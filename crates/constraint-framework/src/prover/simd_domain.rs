use std::ops::Mul;

use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::utils::offset_bit_reversed_circle_domain_index;
use stwo::core::Fraction;
use stwo::prover::backend::simd::column::VeryPackedBaseColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{
    VeryPackedBaseField, VeryPackedSecureField, LOG_N_VERY_PACKED_ELEMS,
};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use crate::logup::LogupAtRow;
use crate::{EvalAtRow, INTERACTION_TRACE_IDX, MAX_N_INTERACTIONS};

/// Evaluates constraints at an evaluation domain points.
pub struct SimdDomainEvaluator<'a> {
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    pub column_index_per_interaction: [usize; MAX_N_INTERACTIONS],
    /// The row index of the simd-vector row to evaluate the constraints at.
    pub vec_row: usize,
    /// Random coefficient powers, pre-broadcast to SIMD lanes. Broadcasting once per
    /// component (instead of per row per constraint) keeps it out of the hot loop.
    pub random_coeff_powers: &'a [VeryPackedSecureField],
    pub row_res: VeryPackedSecureField,
    pub constraint_index: usize,
    pub domain_log_size: u32,
    pub eval_domain_log_size: u32,
    pub logup: LogupAtRow<Self>,
}
impl<'a> SimdDomainEvaluator<'a> {
    /// Broadcasts each random coefficient power to all SIMD lanes, for reuse across all
    /// rows of the evaluation domain.
    pub fn broadcast_random_coeff_powers(
        random_coeff_powers: &[SecureField],
    ) -> Vec<VeryPackedSecureField> {
        random_coeff_powers
            .iter()
            .map(|&p| VeryPackedSecureField::broadcast(p))
            .collect()
    }

    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
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
            vec_row,
            random_coeff_powers,
            row_res: VeryPackedSecureField::zero(),
            constraint_index: 0,
            domain_log_size,
            eval_domain_log_size: eval_log_size,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}
impl EvalAtRow for SimdDomainEvaluator<'_> {
    type F = VeryPackedBaseField;
    type EF = VeryPackedSecureField;

    // TODO(Ohad): Add debug boundary checks.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, we can just return the value directly from this row.
            if off == 0 {
                unsafe {
                    let col = &self
                        .trace_eval
                        .get_unchecked(interaction)
                        .get_unchecked(col_index)
                        .values;
                    let very_packed_col = VeryPackedBaseColumn::transform_under_ref(col);
                    return *very_packed_col.data.get_unchecked(self.vec_row);
                };
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            VeryPackedBaseField::from_array(std::array::from_fn(|i| {
                let row_index = offset_bit_reversed_circle_domain_index(
                    (self.vec_row << (LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS)) + i,
                    self.domain_log_size,
                    self.eval_domain_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index].at(row_index)
            }))
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

    fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
        if self.logup.fracs.is_empty() {
            self.logup.is_finalized = false;
        }
        self.logup.fracs.push(fraction);
    }

    /// Specialized version of [`crate::logup_proxy!`]'s `finalize_logup_batched` that skips
    /// secure-field multiplications by numerators equal to one when summing each batch —
    /// the common case for lookup multiplicities. Emits exactly the same constraints.
    fn finalize_logup_batched(&mut self, batch_size: usize) {
        assert!(!self.logup.is_finalized, "LogupAtRow was already finalized");
        assert!(batch_size > 0, "Batch size must be positive");

        let mut fracs = std::mem::take(&mut self.logup.fracs);
        let n_batches = fracs.len().div_ceil(batch_size);
        assert!(n_batches > 0, "No fractions to finalize");

        let mut prev_col_cumsum = VeryPackedSecureField::zero();

        for (batch_idx, chunk) in fracs.chunks(batch_size).enumerate() {
            let cur_frac = sum_fractions_skipping_one_numerators(chunk);
            if batch_idx + 1 < n_batches {
                // All batches except the last are cumulatively summed in new
                // interaction columns.
                let [cur_cumsum] = self.next_extension_interaction_mask(self.logup.interaction, [0]);
                let diff = cur_cumsum - prev_col_cumsum;
                prev_col_cumsum = cur_cumsum;
                self.add_constraint(diff * cur_frac.denominator - cur_frac.numerator);
            } else {
                let [prev_row_cumsum, cur_cumsum] =
                    self.next_extension_interaction_mask(self.logup.interaction, [-1, 0]);

                let diff = cur_cumsum - prev_row_cumsum - prev_col_cumsum;
                // Instead of checking diff = num / denom, check
                // diff = num / denom - cumsum_shift. This makes
                // (num / denom - cumsum_shift) have sum zero, which makes the constraint
                // uniform - apply on all rows.
                let shifted_diff = diff + self.logup.cumsum_shift;

                self.add_constraint(shifted_diff * cur_frac.denominator - cur_frac.numerator);
            }
        }

        fracs.clear();
        self.logup.fracs = fracs;
        self.logup.is_finalized = true;
    }

    fn finalize_logup(&mut self) {
        self.finalize_logup_batched(1)
    }

    fn finalize_logup_in_pairs(&mut self) {
        self.finalize_logup_batched(2)
    }
}

/// Returns whether all lanes hold the canonical representation of one (coordinate 0 is the
/// integer 1, all other coordinates 0). Numerators built from `EF::one()` always match;
/// semantically-one values in another representation merely fall back to the generic path.
fn is_canonical_one(x: &VeryPackedSecureField) -> bool {
    use std::simd::u32x16;
    let one = u32x16::splat(1);
    let zero = u32x16::splat(0);
    x.0.iter().all(|q| {
        let [c0, c1] = q.0;
        let [a, b] = c0.0;
        let [c, d] = c1.0;
        a.into_simd() == one
            && b.into_simd() == zero
            && c.into_simd() == zero
            && d.into_simd() == zero
    })
}

/// Sums a batch of logup fractions, using a/b + c/d = (ad + cb) / (bd) but skipping the
/// numerator multiplications when a numerator is (bitwise) one. The result is identical to
/// the generic `Fraction` sum.
fn sum_fractions_skipping_one_numerators(
    chunk: &[Fraction<VeryPackedSecureField, VeryPackedSecureField>],
) -> Fraction<VeryPackedSecureField, VeryPackedSecureField> {
    let mut iter = chunk.iter();
    let first = *iter.next().expect("empty logup batch");
    iter.fold(first, |a, b| {
        let lhs = if is_canonical_one(&a.numerator) {
            b.denominator
        } else {
            b.denominator * a.numerator
        };
        let rhs = if is_canonical_one(&b.numerator) {
            a.denominator
        } else {
            a.denominator * b.numerator
        };
        Fraction::new(lhs + rhs, a.denominator * b.denominator)
    })
}
