use std::ops::Mul;

use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::utils::offset_bit_reversed_circle_domain_index;
use stwo::core::Fraction;
use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo::prover::backend::simd::very_packed_m31::{
    Vectorized, VeryPackedBaseField, VeryPackedSecureField, N_VERY_PACKED_ELEMS,
};
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use crate::logup::LogupAtRow;
use crate::{EvalAtRow, INTERACTION_TRACE_IDX, MAX_N_INTERACTIONS};

/// Number of consecutive evaluation-domain rows evaluated per batch.
pub const BATCH_ROWS: usize = N_LANES * N_VERY_PACKED_ELEMS;

/// Evaluates constraints at [`BATCH_ROWS`] consecutive evaluation-domain rows at once on
/// the CPU backend. The lanes are genuine 16-wide packed field elements (the same
/// `VeryPacked` types the SIMD evaluator uses), loaded from the unaligned CPU columns by
/// copy, so all constraint arithmetic runs on the packed kernels rather than relying on
/// scalar auto-vectorization. Produces exactly the values of [`BATCH_ROWS`] single-row
/// [`super::CpuDomainEvaluator`] evaluations.
pub struct BatchCpuDomainEvaluator<'a> {
    pub trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
    pub column_index_per_interaction: [usize; MAX_N_INTERACTIONS],
    /// First row of the batch; scalar lane `k` evaluates row `row + k`.
    pub row: usize,
    /// Random coefficient powers, pre-broadcast to the batch lanes.
    pub random_coeff_powers: &'a [VeryPackedSecureField],
    pub row_res: VeryPackedSecureField,
    pub constraint_index: usize,
    pub domain_log_size: u32,
    pub eval_domain_log_size: u32,
    pub logup: LogupAtRow<Self>,
}

impl<'a> BatchCpuDomainEvaluator<'a> {
    /// Broadcasts each random coefficient power to all batch lanes, for reuse across all
    /// batches of the evaluation domain.
    pub fn broadcast_random_coeff_powers(
        random_coeff_powers: &[SecureField],
    ) -> Vec<VeryPackedSecureField> {
        random_coeff_powers
            .iter()
            .map(|&p| VeryPackedSecureField::broadcast(p))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
        row: usize,
        random_coeff_powers: &'a [VeryPackedSecureField],
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
            row_res: VeryPackedSecureField::zero(),
            constraint_index: 0,
            domain_log_size,
            eval_domain_log_size: eval_log_size,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}

impl EvalAtRow for BatchCpuDomainEvaluator<'_> {
    type F = VeryPackedBaseField;
    type EF = VeryPackedSecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, the lanes read BATCH_ROWS consecutive values from this
            // row on, as unaligned packed copies.
            if off == 0 {
                // Safety: `interaction`, `col_index` and the batch rows are within their
                // respective bounds by construction.
                unsafe {
                    let col = self
                        .trace_eval
                        .get_unchecked(interaction)
                        .get_unchecked(col_index);
                    return Vectorized::from_fn(|j| {
                        let start = self.row + j * N_LANES;
                        PackedM31::from_array(std::array::from_fn(|i| {
                            *col.values.get_unchecked(start + i)
                        }))
                    });
                }
            }
            // Otherwise, each lane looks up the value at the bit-reversed natural order
            // index at an offset.
            Vectorized::from_fn(|j| {
                PackedM31::from_array(std::array::from_fn(|i| {
                    let row = offset_bit_reversed_circle_domain_index(
                        self.row + j * N_LANES + i,
                        self.domain_log_size,
                        self.eval_domain_log_size,
                        off,
                    );
                    self.trace_eval[interaction][col_index][row]
                }))
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
        VeryPackedSecureField::from_very_packed_m31s(values)
    }

    crate::logup_proxy!();
}

/// Loads the four coordinate columns of a secure column at `row` as packed lanes.
pub fn load_secure_coords(
    columns: &[Vec<BaseField>; SECURE_EXTENSION_DEGREE],
    row: usize,
) -> PackedQM31 {
    PackedQM31::from_packed_m31s(std::array::from_fn(|c| {
        PackedM31::from_array(std::array::from_fn(|i| columns[c][row + i]))
    }))
}
