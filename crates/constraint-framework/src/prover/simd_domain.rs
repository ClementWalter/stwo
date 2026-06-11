use std::ops::Mul;

use num_traits::{One, Zero};
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
    /// Logup fraction denominators, in entry order. Numerators are stored sparsely in
    /// [`Self::logup_nonunit_numerators`]; an entry without one has a numerator of one
    /// (the common case for lookup multiplicities), avoiding a 128-byte copy per entry
    /// and the multiplications by one when batches are summed.
    pub logup_denoms: Vec<VeryPackedSecureField>,
    /// Sparse `(entry index, numerator)` pairs for entries whose numerator is not
    /// (bitwise) the canonical one, in increasing entry order.
    pub logup_nonunit_numerators: Vec<(usize, VeryPackedSecureField)>,
    /// Precomputed `(offset, index map)` pairs for non-zero mask offsets: the map gives,
    /// for every flat evaluation index, the bit-reversed circle-domain index of its offset
    /// neighbor. Offsets without a map fall back to computing the index per lane.
    pub offset_index_maps: &'a [(isize, Vec<u32>)],
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
            logup_denoms: Vec::new(),
            logup_nonunit_numerators: Vec::new(),
            offset_index_maps: &[],
        }
    }

    fn push_logup_frac(
        &mut self,
        numerator: Option<VeryPackedSecureField>,
        denominator: VeryPackedSecureField,
    ) {
        if self.logup_denoms.is_empty() {
            self.logup.is_finalized = false;
        }
        if let Some(numerator) = numerator {
            self.logup_nonunit_numerators
                .push((self.logup_denoms.len(), numerator));
        }
        self.logup_denoms.push(denominator);
    }

    /// Sums the logup fraction batch with entry indices `[start, end)`, skipping the
    /// numerator multiplications for entries whose numerator is one. `numerators` is the
    /// sparse non-unit numerator suffix that starts at `start` (or later). Returns the batch
    /// fraction's numerator (`None` encodes one) and denominator, plus the remaining sparse
    /// suffix. The result is identical to the generic `Fraction` sum.
    fn sum_logup_batch<'b>(
        denoms: &[VeryPackedSecureField],
        mut numerators: &'b [(usize, VeryPackedSecureField)],
        start: usize,
        end: usize,
    ) -> (
        Option<VeryPackedSecureField>,
        VeryPackedSecureField,
        &'b [(usize, VeryPackedSecureField)],
    ) {
        let mut numerator_at = |i: usize| -> Option<VeryPackedSecureField> {
            match numerators.first() {
                Some(&(idx, n)) if idx == i => {
                    numerators = &numerators[1..];
                    Some(n)
                }
                _ => None,
            }
        };

        let mut acc_num = numerator_at(start);
        let mut acc_den = denoms[start];
        for (i, &den) in denoms.iter().enumerate().take(end).skip(start + 1) {
            let num = numerator_at(i);
            // a/b + c/d = (ad + cb) / (bd), with multiplications by one elided.
            let lhs = match acc_num {
                None => den,
                Some(a) => den * a,
            };
            let rhs = match num {
                None => acc_den,
                Some(c) => acc_den * c,
            };
            acc_num = Some(lhs + rhs);
            acc_den *= den;
        }
        (acc_num, acc_den, numerators)
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
            let base = self.vec_row << (LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS);
            let index_map = self
                .offset_index_maps
                .iter()
                .find(|(map_off, _)| *map_off == off)
                .map(|(_, map)| map);
            VeryPackedBaseField::from_array(std::array::from_fn(|i| {
                let row_index = match index_map {
                    Some(map) => map[base + i] as usize,
                    None => offset_bit_reversed_circle_domain_index(
                        base + i,
                        self.domain_log_size,
                        self.eval_domain_log_size,
                        off,
                    ),
                };
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

    /// Stores the relation entry in compact form: the denominator is pushed densely and the
    /// numerator only when it is not (bitwise) the canonical one. This avoids materializing
    /// and copying a full `Fraction` per entry in the hot row loop.
    fn add_to_relation<R: crate::Relation<Self::F, Self::EF>>(
        &mut self,
        entry: crate::RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let denominator = entry.relation.combine(entry.values);
        let numerator = if is_canonical_one(&entry.multiplicity) {
            None
        } else {
            Some(entry.multiplicity)
        };
        self.push_logup_frac(numerator, denominator);
    }

    fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
        let numerator = if is_canonical_one(&fraction.numerator) {
            None
        } else {
            Some(fraction.numerator)
        };
        self.push_logup_frac(numerator, fraction.denominator);
    }

    /// Specialized version of [`crate::logup_proxy!`]'s `finalize_logup_batched` that reads
    /// the compact fraction storage and skips secure-field multiplications by numerators
    /// equal to one — the common case for lookup multiplicities. Emits exactly the same
    /// constraints as the generic implementation.
    fn finalize_logup_batched(&mut self, batch_size: usize) {
        assert!(!self.logup.is_finalized, "LogupAtRow was already finalized");
        assert!(batch_size > 0, "Batch size must be positive");

        let mut denoms = std::mem::take(&mut self.logup_denoms);
        let mut nonunit_numerators = std::mem::take(&mut self.logup_nonunit_numerators);
        let n_batches = denoms.len().div_ceil(batch_size);
        assert!(n_batches > 0, "No fractions to finalize");

        let mut prev_col_cumsum = VeryPackedSecureField::zero();
        let mut numerators = nonunit_numerators.as_slice();

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(denoms.len());
            let (cur_num, cur_den, rest) = Self::sum_logup_batch(&denoms, numerators, start, end);
            numerators = rest;
            if batch_idx + 1 < n_batches {
                // All batches except the last are cumulatively summed in new
                // interaction columns.
                let [cur_cumsum] =
                    self.next_extension_interaction_mask(self.logup.interaction, [0]);
                let diff = cur_cumsum - prev_col_cumsum;
                prev_col_cumsum = cur_cumsum;
                let lhs = diff * cur_den;
                self.add_constraint(match cur_num {
                    None => lhs - VeryPackedSecureField::one(),
                    Some(num) => lhs - num,
                });
            } else {
                let [prev_row_cumsum, cur_cumsum] =
                    self.next_extension_interaction_mask(self.logup.interaction, [-1, 0]);

                let diff = cur_cumsum - prev_row_cumsum - prev_col_cumsum;
                // Instead of checking diff = num / denom, check
                // diff = num / denom - cumsum_shift. This makes
                // (num / denom - cumsum_shift) have sum zero, which makes the constraint
                // uniform - apply on all rows.
                let shifted_diff = diff + self.logup.cumsum_shift;

                let lhs = shifted_diff * cur_den;
                self.add_constraint(match cur_num {
                    None => lhs - VeryPackedSecureField::one(),
                    Some(num) => lhs - num,
                });
            }
        }

        denoms.clear();
        nonunit_numerators.clear();
        self.logup_denoms = denoms;
        self.logup_nonunit_numerators = nonunit_numerators;
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
/// semantically-one values in another representation merely take the generic path.
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
