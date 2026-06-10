use std::ops::Mul;

use num_traits::Zero;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::{SecureField, QM31, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::utils::offset_bit_reversed_circle_domain_index;
use stwo::core::Fraction;
use stwo::prover::backend::simd::very_packed_m31::Vectorized;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use crate::logup::LogupAtRow;
use crate::{EvalAtRow, INTERACTION_TRACE_IDX, MAX_N_INTERACTIONS};

/// Evaluates constraints at `K` consecutive evaluation-domain rows at once on the CPU
/// backend. Batching rows breaks the per-row dependency chains of deep constraint
/// expressions (e.g. iterated squarings) across independent lanes, and the elementwise
/// lane arithmetic auto-vectorizes. Produces exactly the values of `K` single-row
/// [`super::CpuDomainEvaluator`] evaluations.
pub struct BatchCpuDomainEvaluator<'a, const K: usize> {
    pub trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
    pub column_index_per_interaction: [usize; MAX_N_INTERACTIONS],
    /// First row of the batch; lane `k` evaluates row `row + k`.
    pub row: usize,
    /// Random coefficient powers, pre-broadcast to the batch lanes.
    pub random_coeff_powers: &'a [Vectorized<QM31, K>],
    pub row_res: Vectorized<QM31, K>,
    pub constraint_index: usize,
    pub domain_log_size: u32,
    pub eval_domain_log_size: u32,
    pub logup: LogupAtRow<Self>,
}

impl<'a, const K: usize> BatchCpuDomainEvaluator<'a, K> {
    /// Broadcasts each random coefficient power to all batch lanes, for reuse across all
    /// batches of the evaluation domain.
    pub fn broadcast_random_coeff_powers(
        random_coeff_powers: &[SecureField],
    ) -> Vec<Vectorized<QM31, K>> {
        random_coeff_powers
            .iter()
            .map(|&p| Vectorized::from_fn(|_| p))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
        row: usize,
        random_coeff_powers: &'a [Vectorized<QM31, K>],
        domain_log_size: u32,
        eval_log_size: u32,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace_eval,
            column_index_per_interaction: {
                debug_assert!(trace_eval.len() <= MAX_N_INTERACTIONS);
                [0; MAX_N_INTERACTIONS]
            },
            row,
            random_coeff_powers,
            row_res: Vectorized::from_fn(|_| QM31::zero()),
            constraint_index: 0,
            domain_log_size,
            eval_domain_log_size: eval_log_size,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}

impl<const K: usize> EvalAtRow for BatchCpuDomainEvaluator<'_, K> {
    type F = Vectorized<M31, K>;
    type EF = Vectorized<QM31, K>;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, the lanes read K consecutive values from this row on.
            if off == 0 {
                // Safety: `interaction`, `col_index` and the batch rows are within their
                // respective bounds by construction.
                unsafe {
                    let col = self
                        .trace_eval
                        .get_unchecked(interaction)
                        .get_unchecked(col_index);
                    return Vectorized::from_fn(|k| *col.values.get_unchecked(self.row + k));
                }
            }
            // Otherwise, each lane looks up the value at the bit-reversed natural order
            // index at an offset.
            Vectorized::from_fn(|k| {
                let row = offset_bit_reversed_circle_domain_index(
                    self.row + k,
                    self.domain_log_size,
                    self.eval_domain_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index][row]
            })
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        self.row_res += self.random_coeff_powers[self.constraint_index] * constraint;
        self.constraint_index += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        Vectorized::from_fn(|k| {
            QM31::from_m31_array([
                values[0].0[k],
                values[1].0[k],
                values[2].0[k],
                values[3].0[k],
            ])
        })
    }

    crate::logup_proxy!();
}
