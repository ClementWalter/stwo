//! Parametric AIR for prover-throughput benchmarking by trace shape.
//!
//! The trace is an `n_cols`-wide grid of M31 columns over `2^log_n_rows` rows:
//! the first two columns hold per-row seeds and every later column is the
//! wide-Fibonacci step `c = a^2 + b^2` of the two before it. Two constraint
//! modes share this trace:
//!
//! - **constrained** — one degree-2 constraint per derived column, so constraint evaluation scales
//!   with the trace like a real AIR;
//! - **unconstrained** — the columns are only committed (masks without constraints), measuring the
//!   commitment/FRI envelope alone, in the spirit of rookie-numbers' "theoretical maximum
//!   frequency" benchmark.
//!
//! The shape is chosen at runtime (no const generics) so a single binary can
//! sweep a (rows x cols) grid, and proofs stay bit-comparable across builds.

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedBaseField, N_LANES};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

pub type AirShapeComponent = FrameworkComponent<AirShapeEval>;

#[derive(Clone)]
pub struct AirShapeEval {
    pub log_n_rows: u32,
    pub n_cols: usize,
    pub constrained: bool,
}

impl FrameworkEval for AirShapeEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        if !self.constrained {
            for _ in 0..self.n_cols {
                eval.next_trace_mask();
            }
            return eval;
        }
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..self.n_cols {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.clone() * a.clone() + b.clone() * b.clone()));
            a = b;
            b = c;
        }
        eval
    }
}

/// Generates the trace: per-row seeds `(1, row_index)` followed by the
/// `a^2 + b^2` chain. Each SIMD lane carries one row, so column writes are
/// contiguous.
pub fn generate_trace(
    log_n_rows: u32,
    n_cols: usize,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    assert!(n_cols >= 2);
    let n_rows = 1usize << log_n_rows;
    let mut trace = (0..n_cols)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(n_rows))
        .collect_vec();

    let chunk_rows = (1usize << 14).min(n_rows);
    let n_chunks = n_rows / chunk_rows;
    let mut col_chunks = trace
        .iter_mut()
        .map(|c| c.data.chunks_mut(chunk_rows / N_LANES))
        .collect_vec();
    let mut chunk_views = (0..n_chunks)
        .map(|i| {
            (
                i * chunk_rows,
                col_chunks
                    .iter_mut()
                    .map(|it| it.next().unwrap())
                    .collect_vec(),
            )
        })
        .collect_vec();

    let process_chunk = |(start_row, cols): &mut (usize, Vec<&mut [PackedBaseField]>)| {
        for vec_idx in 0..cols[0].len() {
            let row0 = *start_row + vec_idx * N_LANES;
            let mut a = PackedBaseField::broadcast(BaseField::from(1));
            let mut b = PackedBaseField::from_array(std::array::from_fn(|k| {
                BaseField::from((row0 + k) as u32)
            }));
            cols[0][vec_idx] = a;
            cols[1][vec_idx] = b;
            for col in cols.iter_mut().skip(2) {
                (a, b) = (b, a * a + b * b);
                col[vec_idx] = b;
            }
        }
    };

    #[cfg(feature = "parallel")]
    chunk_views.par_iter_mut().for_each(process_chunk);
    #[cfg(not(feature = "parallel"))]
    chunk_views.iter_mut().for_each(process_chunk);

    let domain = CanonicCoset::new(log_n_rows).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sM31Channel;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
    use stwo::core::verifier::verify;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::{prove, CommitmentSchemeProver};
    use stwo_constraint_framework::TraceLocationAllocator;

    use super::{generate_trace, AirShapeComponent, AirShapeEval};

    /// Proves one (rows, cols) shape, timing the protocol (twiddles through
    /// proof) while excluding trace generation, and prints a machine-parsable
    /// `AIR_SHAPE ...` line with the cell throughput. Shape comes from env:
    ///
    ///   LOG_N_ROWS=18 N_COLS=64 CONSTRAINED=1 \
    ///     cargo test --release -p stwo-examples --features parallel \
    ///     test_air_shape_prove -- --nocapture
    #[test_log::test]
    fn test_air_shape_prove() {
        let log_n_rows: u32 = std::env::var("LOG_N_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14);
        let n_cols: usize = std::env::var("N_COLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let constrained = std::env::var("CONSTRAINED").map_or(true, |v| v != "0");

        let trace = generate_trace(log_n_rows, n_cols);

        let start = std::time::Instant::now();
        let config = PcsConfig::default();
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_n_rows + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        let component = AirShapeComponent::new(
            &mut TraceLocationAllocator::default(),
            AirShapeEval {
                log_n_rows,
                n_cols,
                constrained,
            },
            SecureField::zero(),
        );

        let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();
        let prove_s = start.elapsed().as_secs_f64();

        let cells = (1u64 << log_n_rows) * n_cols as u64;
        println!(
            "AIR_SHAPE log_rows={log_n_rows} cols={n_cols} constrained={} prove_s={prove_s:.4} \
             mcells_s={:.2}",
            constrained as u8,
            cells as f64 / prove_s / 1e6,
        );
        crate::maybe_dump_proof_hash("air_shape", &proof);

        let verifier_channel = &mut Blake2sM31Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
        commitment_scheme.commit(proof.commitments[0], &[], verifier_channel);
        commitment_scheme.commit(
            proof.commitments[1],
            &component.trace_log_degree_bounds()[1],
            verifier_channel,
        );
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }
}
