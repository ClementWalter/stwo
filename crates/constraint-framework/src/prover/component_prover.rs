use std::borrow::Cow;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::air::Component;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
use stwo::core::utils::{bit_reverse, offset_bit_reversed_circle_domain_index};
use stwo::prover::backend::simd::column::VeryPackedSecureColumnByCoords;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Backend, CpuBackend};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{ComponentProver, DomainEvaluationAccumulator, EvaluationMode, Poly, Trace};
use tracing::{span, Level};

use super::{BatchCpuDomainEvaluator, CpuDomainEvaluator, SimdDomainEvaluator};
use crate::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

/// Number of very-packed rows evaluated per rayon task. Large enough to amortize
/// work-stealing overhead, small enough to balance load across threads.
const CHUNK_SIZE: usize = 32;

/// Common inputs for constraint quotient evaluation, shared between the SIMD and CPU backends.
struct ConstraintQuotientInputs<'a, B: Backend> {
    eval_domain: CircleDomain,
    trace_domain: CanonicCoset,
    trace: TreeVec<Vec<Cow<'a, CircleEvaluation<B, BaseField, BitReversedOrder>>>>,
    denom_inv: Vec<BaseField>,
}

/// Prepares trace evaluations: borrows directly (subdomain) or extends to eval domain.
fn get_trace_columns<'a, B: Backend>(
    component_polys: TreeVec<Vec<&'a &Poly<B>>>,
    eval_domain: CircleDomain,
    mode: EvaluationMode,
) -> TreeVec<Vec<Cow<'a, CircleEvaluation<B, BaseField, BitReversedOrder>>>> {
    match mode {
        EvaluationMode::SubDomain { .. } => {
            // Borrow committed evaluations directly. Only the first
            // 2^max_constraint_log_degree_bound indices are going to be used for the
            // constraint quotient evaluation (in bit-reversed order these form the
            // subdomain coset).
            //
            // Ideally we'd slice to just those indices, but the type system requires
            // borrowing the entire evaluation.
            component_polys.map_cols(|c| Cow::Borrowed(&c.evals))
        }
        EvaluationMode::ExtendToEvalDomain => {
            let _span = span!(Level::INFO, "Constraint Extension").entered();
            let twiddles = B::precompute_twiddles(eval_domain.half_coset);
            #[cfg(not(feature = "parallel"))]
            {
                component_polys.as_cols_ref().map_cols(|col| {
                    Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
                })
            }
            #[cfg(feature = "parallel")]
            {
                component_polys.as_cols_ref().par_map_cols(|col| {
                    Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
                })
            }
        }
    }
}

/// Constructs the inputs needed for constraint quotient evaluation from a component and trace.
/// Computes the eval/trace domains, prepares trace columns (borrowing or extending as needed),
/// and precomputes denominator inverses.
fn get_constraint_quotient_inputs<'a, E: FrameworkEval, B: Backend>(
    component: &FrameworkComponent<E>,
    trace: &'a Trace<'a, B>,
    mode: EvaluationMode,
) -> ConstraintQuotientInputs<'a, B> {
    let max_constraint_log_degree_bound = component.max_constraint_log_degree_bound();
    let trace_domain = CanonicCoset::new(component.eval.log_size());

    let mut component_polys = trace.polys.sub_tree(&component.trace_locations);
    component_polys[PREPROCESSED_TRACE_IDX] = component
        .preprocessed_column_indices
        .iter()
        .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
        .collect();

    let eval_domain = match mode {
        EvaluationMode::SubDomain { log_expansion } => {
            subdomain_eval_domain(max_constraint_log_degree_bound, log_expansion)
        }
        EvaluationMode::ExtendToEvalDomain => {
            CanonicCoset::new(max_constraint_log_degree_bound).circle_domain()
        }
    };
    let trace = get_trace_columns(component_polys, eval_domain, mode);

    // Denom inverses.
    let log_expand = eval_domain.log_size() - trace_domain.log_size();
    let mut denom_inv = (0..1 << log_expand)
        .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
        .collect_vec();
    bit_reverse(&mut denom_inv);

    ConstraintQuotientInputs {
        eval_domain,
        trace_domain,
        trace,
        denom_inv,
    }
}

impl<E: FrameworkEval + Sync> ComponentProver<SimdBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }

        let ConstraintQuotientInputs {
            eval_domain,
            trace_domain,
            trace,
            denom_inv,
        } = get_constraint_quotient_inputs(self, trace, evaluation_accumulator.evaluation_mode());

        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "Constraint point-wise eval",
            class = "ConstraintEval"
        )
        .entered();

        // Fall back to CPU if the trace is too small.
        if trace_domain.log_size() < LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS {
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.to_cpu());
            let trace_cols = trace_cols.as_cols_ref();
            *accum.col = SecureColumnByCoords::from_cpu(accumulate_pointwise_cpu(
                &self.eval,
                self.claimed_sum,
                trace_cols,
                eval_domain.log_size(),
                trace_domain.log_size(),
                denom_inv,
                &accum.random_coeff_powers,
                &accum.col.to_cpu(),
            ));
            return;
        }

        let col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(accum.col) };

        // Number of valid very-packed rows. The transformed column's inner vectors still
        // report their pre-transform (PackedBaseField) length, so chunks taken from them
        // can extend past this count and must be clamped.
        let n_vec_rows = 1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS);
        let range = 0..n_vec_rows;

        #[cfg(not(feature = "parallel"))]
        let iter = range.step_by(CHUNK_SIZE).zip(col.chunks_mut(CHUNK_SIZE));

        #[cfg(feature = "parallel")]
        let iter = range
            .into_par_iter()
            .step_by(CHUNK_SIZE)
            .zip(col.par_chunks_mut(CHUNK_SIZE));

        // Define any `self` values outside the loop to prevent the compiler thinking there is a
        // `Sync` requirement on `Self`.
        let self_eval = &self.eval;
        let self_claimed_sum = self.claimed_sum;

        // Shared read-only view of the trace columns, built once and borrowed by every
        // row task to avoid per-row allocations inside the hot loop.
        let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());
        let trace_cols = &trace_cols;
        // Broadcast the random coefficient powers to SIMD lanes once for all rows.
        let broadcast_powers =
            SimdDomainEvaluator::broadcast_random_coeff_powers(&accum.random_coeff_powers);
        let broadcast_powers = &broadcast_powers;
        // Precompute, for every non-zero mask offset, the bit-reversed circle-domain index
        // of each evaluation point's offset neighbor. The maps are shared by all rows and
        // all columns using that offset, replacing per-row per-lane index computations.
        let offset_index_maps: Vec<(isize, Vec<u32>)> = self
            .nonzero_mask_offsets()
            .into_iter()
            .map(|off| {
                let n = 1usize << eval_domain.log_size();
                let mut map = vec![0u32; n];
                #[cfg(feature = "parallel")]
                let chunks = map.par_chunks_mut(1 << 12);
                #[cfg(not(feature = "parallel"))]
                let chunks = map.chunks_mut(1 << 12);
                chunks.enumerate().for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx << 12;
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        *slot = offset_bit_reversed_circle_domain_index(
                            base + j,
                            trace_domain.log_size(),
                            eval_domain.log_size(),
                            off,
                        ) as u32;
                    }
                });
                (off, map)
            })
            .collect();
        let offset_index_maps = &offset_index_maps;

        iter.for_each(|(chunk_start_row, mut chunk)| {
            // Logup fraction buffers, recycled across the chunk's rows: finalize_logup_batched
            // hands the vectors back cleared, so only the first row of the chunk allocates.
            let mut denoms_buf = Vec::new();
            let mut numerators_buf = Vec::new();
            // Clamp to both the chunk length and the valid row count (see n_vec_rows above).
            let chunk_rows = chunk.0[0].0.len().min(n_vec_rows - chunk_start_row);
            for idx_in_chunk in 0..chunk_rows {
                let vec_row = chunk_start_row + idx_in_chunk;
                // Evaluate constrains at row.
                let mut eval = SimdDomainEvaluator::new(
                    trace_cols,
                    vec_row,
                    broadcast_powers,
                    trace_domain.log_size(),
                    eval_domain.log_size(),
                    self_eval.log_size(),
                    self_claimed_sum,
                );
                eval.offset_index_maps = offset_index_maps;
                eval.logup_denoms = std::mem::take(&mut denoms_buf);
                eval.logup_nonunit_numerators = std::mem::take(&mut numerators_buf);
                let mut evaluated = self_eval.evaluate(eval);
                let row_res = evaluated.row_res;
                denoms_buf = std::mem::take(&mut evaluated.logup_denoms);
                numerators_buf = std::mem::take(&mut evaluated.logup_nonunit_numerators);

                // Finalize row.
                unsafe {
                    let row_denom_inv = VeryPackedBaseField::broadcast(
                        denom_inv[vec_row
                            >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
                    );
                    chunk.set_packed(
                        idx_in_chunk,
                        chunk.packed_at(idx_in_chunk) + row_res * row_denom_inv,
                    )
                }
            }
        });
    }
}

impl<E: FrameworkEval + Sync> ComponentProver<CpuBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }

        let ConstraintQuotientInputs {
            eval_domain,
            trace_domain,
            trace,
            denom_inv,
        } = get_constraint_quotient_inputs(self, trace, evaluation_accumulator.evaluation_mode());

        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "Constraint point-wise eval",
            class = "ConstraintEval"
        )
        .entered();
        let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

        *accum.col = accumulate_pointwise_cpu(
            &self.eval,
            self.claimed_sum,
            trace_cols,
            eval_domain.log_size(),
            trace_domain.log_size(),
            denom_inv,
            &accum.random_coeff_powers,
            accum.col,
        );
    }
}

/// Computes the evaluation subdomain for a component given its constraint degree bound
/// and the log_expansion from `EvaluationMode::SubDomain`.
///
/// When `log_expansion == 0`, returns the canonical domain.
/// When `log_expansion > 0`, returns the first subdomain obtained by splitting the
/// committed domain `log_expansion` times.
fn subdomain_eval_domain(max_constraint_log_degree_bound: u32, log_expansion: u32) -> CircleDomain {
    let committed_domain =
        CanonicCoset::new(max_constraint_log_degree_bound + log_expansion).circle_domain();
    committed_domain.split(log_expansion).0
}

#[allow(clippy::too_many_arguments)]
fn accumulate_pointwise_cpu<E: FrameworkEval + Sync>(
    component_eval: &E,
    claimed_sum: SecureField,
    trace_cols: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
    eval_log_size: u32,
    trace_log_size: u32,
    denom_inv: Vec<BaseField>,
    random_coeff_powers: &[SecureField],
    accum: &SecureColumnByCoords<CpuBackend>,
) -> SecureColumnByCoords<CpuBackend> {
    let mut res = SecureColumnByCoords::zeros(1 << eval_log_size);

    // Rows are independent; evaluate disjoint row chunks concurrently (each chunk owns
    // a disjoint range of every coordinate column), and within a chunk evaluate BATCH
    // consecutive rows at a time: lanes break the per-row dependency chains of deep
    // constraint expressions.
    const BATCH: usize = 8;
    let chunk_size = 1 << 12;
    let trace_cols = &trace_cols;
    let denom_inv = &denom_inv;
    let batch_powers =
        BatchCpuDomainEvaluator::<BATCH>::broadcast_random_coeff_powers(random_coeff_powers);
    let batch_powers = &batch_powers;
    let mut chunk_views = {
        let [c0, c1, c2, c3]: &mut [Vec<BaseField>; 4] = &mut res.columns;
        (c0.chunks_mut(chunk_size))
            .zip(c1.chunks_mut(chunk_size))
            .zip(c2.chunks_mut(chunk_size))
            .zip(c3.chunks_mut(chunk_size))
            .enumerate()
            .map(|(i, (((d0, d1), d2), d3))| (i * chunk_size, [d0, d1, d2, d3]))
            .collect_vec()
    };

    let process_chunk = |(start, chunk): &mut (usize, [&mut [BaseField]; 4])| {
        let rows = chunk[0].len();
        let mut idx = 0;
        while idx + BATCH <= rows {
            let row = *start + idx;
            // Evaluate constraints at rows row..row + BATCH.
            let eval = BatchCpuDomainEvaluator::<BATCH>::new(
                trace_cols,
                row,
                batch_powers,
                trace_log_size,
                eval_log_size,
                component_eval.log_size(),
                claimed_sum,
            );
            let row_res = component_eval.evaluate(eval).row_res;

            // Finalize the batch.
            for (k, lane_res) in row_res.0.into_iter().enumerate() {
                let lane_row = row + k;
                let row_denom_inv = denom_inv[lane_row >> trace_log_size];
                let [v0, v1, v2, v3] =
                    (accum.at(lane_row) + lane_res * row_denom_inv).to_m31_array();
                chunk[0][idx + k] = v0;
                chunk[1][idx + k] = v1;
                chunk[2][idx + k] = v2;
                chunk[3][idx + k] = v3;
            }
            idx += BATCH;
        }
        // Tail rows shorter than a batch.
        for idx in idx..rows {
            let row = *start + idx;
            // Evaluate constrains at row.
            let eval = CpuDomainEvaluator::new(
                trace_cols,
                row,
                random_coeff_powers,
                trace_log_size,
                eval_log_size,
                component_eval.log_size(),
                claimed_sum,
            );
            let row_res = component_eval.evaluate(eval).row_res;

            // Finalize row.
            let row_denom_inv = denom_inv[row >> trace_log_size];
            let [v0, v1, v2, v3] = (accum.at(row) + row_res * row_denom_inv).to_m31_array();
            chunk[0][idx] = v0;
            chunk[1][idx] = v1;
            chunk[2][idx] = v2;
            chunk[3][idx] = v3;
        }
    };

    #[cfg(feature = "parallel")]
    chunk_views.par_iter_mut().for_each(process_chunk);
    #[cfg(not(feature = "parallel"))]
    chunk_views.iter_mut().for_each(process_chunk);

    res
}
