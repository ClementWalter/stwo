use itertools::{multizip, Itertools};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::uninit_vec;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::cm31::PackedCM31;
use stwo::prover::backend::simd::column::SecureColumn;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::prefix_sum::inclusive_prefix_sum;
use stwo::prover::backend::simd::qm31::{batch_inverse_packed_qm31, PackedQM31, PackedSecureField};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;

/// Fails closed before a LogUp denominator can reach packed inversion.
///
/// LogUp fractions are defined only away from their poles. A packed field
/// element may contain a zero in just one SIMD lane, so `PackedQM31::is_zero`
/// is not sufficient here: it reports whether every lane is zero. `to_array`
/// canonicalizes each lane first, making raw `P` coordinates indistinguishable
/// from the other M31 zero representative (`0`). This check is unconditional in
/// debug and release builds.
#[track_caller]
#[inline]
fn assert_no_logup_poles(denom: PackedSecureField) {
    assert!(
        denom
            .to_array()
            .into_iter()
            .all(|lane| lane != SecureField::zero()),
        "logup denominator contains a zero lane"
    );
}

// SIMD backend generator for logup interaction trace.
pub struct LogupTraceGenerator {
    log_size: u32,
    /// Current allocated interaction columns.
    trace: Vec<SecureColumnByCoords<SimdBackend>>,
    /// Denominator expressions (z + sum_i alpha^i * x_i) being generated for the current lookup.
    denom: SecureColumn,
    batch_inverse_buffer: Vec<PackedSecureField>,
}
impl LogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureColumn::zeros(1 << log_size);
        let batch_inverse_buffer = unsafe { uninit_vec(1 << (log_size - LOG_N_LANES)) };
        Self {
            log_size,
            trace,
            denom,
            batch_inverse_buffer,
        }
    }

    /// # Safety
    ///
    /// Calling `finalize_last` on uninitialized LogupTraceGenerator leads to undefined behavior.
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureColumn::uninitialized(1 << log_size);
        let batch_inverse_buffer = unsafe { uninit_vec(1 << (log_size - LOG_N_LANES)) };
        Self {
            log_size,
            trace,
            denom,
            batch_inverse_buffer,
        }
    }

    /// Allocate a new lookup column.
    pub fn new_col(&mut self) -> LogupColGenerator<'_> {
        let log_size = self.log_size;
        LogupColGenerator {
            gen: self,
            numerator: unsafe { SecureColumnByCoords::<SimdBackend>::uninitialized(1 << log_size) },
        }
    }

    pub fn col_from_iter(
        &mut self,
        iter: impl ExactSizeIterator<Item = (PackedSecureField, PackedSecureField)>,
    ) {
        let length = 1 << self.log_size;
        assert_eq!(iter.len() * N_LANES, length);
        let mut col_gen = self.new_col();
        for (vec_row, (numerator, denom)) in iter.enumerate() {
            col_gen.write_frac(vec_row, numerator, denom);
        }
        col_gen.finalize_col();
    }

    #[cfg(feature = "parallel")]
    pub fn col_from_par_iter(
        &mut self,
        iter: impl IndexedParallelIterator<Item = (PackedSecureField, PackedSecureField)>,
    ) {
        let length = 1 << self.log_size;
        assert_eq!(iter.len() * N_LANES, length);

        // Write fractions straight into a preallocated column instead of unzipping into
        // four freshly collected vectors.
        let mut numerator = unsafe { SecureColumnByCoords::<SimdBackend>::uninitialized(length) };
        let [n0, n1, n2, n3] = &mut numerator.columns;
        (self.denom.data.par_iter_mut())
            .zip(n0.data.par_iter_mut())
            .zip(n1.data.par_iter_mut())
            .zip(n2.data.par_iter_mut())
            .zip(n3.data.par_iter_mut())
            .zip(iter)
            .for_each(|(((((dst_denom, d0), d1), d2), d3), (numerator, denom))| {
                assert_no_logup_poles(denom);
                *dst_denom = denom;
                let [c0, c1, c2, c3] = numerator.into_packed_m31s();
                *d0 = c0;
                *d1 = c1;
                *d2 = c2;
                *d3 = c3;
            });

        LogupColGenerator {
            gen: self,
            numerator,
        }
        .finalize_col();
    }

    /// Generates `n_cols` lookup columns at once from `frac_at(col, vec_row) ->
    /// (numerator, denominator)`.
    ///
    /// Parallelizes over contiguous row bands instead of per-column passes: each band
    /// computes, for every column in order, the fractions, a band-local batch inverse,
    /// the `numerator * denom^-1` values and the running cross-column sums. This is a
    /// single fork-join with cache-hot chaining, instead of three parallel passes per
    /// column. The resulting columns are identical to `n_cols` successive
    /// `col_from_iter` calls.
    pub fn cols_from_fn(
        &mut self,
        n_cols: usize,
        frac_at: impl Fn(usize, usize) -> (PackedSecureField, PackedSecureField) + Sync,
    ) {
        if n_cols == 0 {
            return;
        }
        let length = 1usize << self.log_size;
        let packed_len = length >> LOG_N_LANES;
        // Band-local batch inversion keeps one field inversion per band per column while
        // the buffers (BAND * 64B) stay cache resident.
        const BAND: usize = 1 << 9;

        let prev_col = self.trace.last();

        let mut new_cols: Vec<SecureColumnByCoords<SimdBackend>> = (0..n_cols)
            .map(|_| unsafe { SecureColumnByCoords::<SimdBackend>::uninitialized(length) })
            .collect();

        // Split every new column into disjoint per-band coordinate slices.
        let mut col_band_iters = new_cols
            .iter_mut()
            .map(|c| {
                let [a, b, cc, d] = &mut c.columns;
                [
                    a.data.chunks_mut(BAND),
                    b.data.chunks_mut(BAND),
                    cc.data.chunks_mut(BAND),
                    d.data.chunks_mut(BAND),
                ]
            })
            .collect_vec();
        let n_bands = packed_len.div_ceil(BAND);
        let mut bands: Vec<(usize, Vec<[&mut [PackedBaseField]; 4]>)> = (0..n_bands)
            .map(|band_idx| {
                (
                    band_idx * BAND,
                    col_band_iters
                        .iter_mut()
                        .map(|its| its.each_mut().map(|it| it.next().unwrap()))
                        .collect(),
                )
            })
            .collect();

        let process_band = |(band_start, cols): &mut (usize, Vec<[&mut [PackedBaseField]; 4]>)| {
            let band_start = *band_start;
            let band_len = cols[0][0].len();
            let mut denoms = vec![PackedSecureField::zero(); band_len];
            let mut denom_invs = vec![PackedSecureField::zero(); band_len];
            // Running cross-column sums for this band's rows, seeded from the previous
            // existing column if any.
            let mut acc: Vec<PackedSecureField> = match prev_col {
                Some(col) => (0..band_len)
                    .map(|i| unsafe { col.packed_at(band_start + i) })
                    .collect(),
                None => vec![PackedSecureField::zero(); band_len],
            };

            for (col_idx, col) in cols.iter_mut().enumerate() {
                for (i, denom_slot) in denoms.iter_mut().enumerate() {
                    let (numerator, denom) = frac_at(col_idx, band_start + i);
                    assert_no_logup_poles(denom);
                    *denom_slot = denom;
                    // Stash the numerator in the output slots until the inverse pass.
                    let [c0, c1, c2, c3] = numerator.into_packed_m31s();
                    col[0][i] = c0;
                    col[1][i] = c1;
                    col[2][i] = c2;
                    col[3][i] = c3;
                }
                batch_inverse_packed_qm31(&denoms, &mut denom_invs);
                for (i, inv) in denom_invs.iter().enumerate() {
                    let numerator = PackedQM31([
                        PackedCM31([col[0][i], col[1][i]]),
                        PackedCM31([col[2][i], col[3][i]]),
                    ]);
                    let value = numerator * *inv + acc[i];
                    acc[i] = value;
                    let [c0, c1, c2, c3] = value.into_packed_m31s();
                    col[0][i] = c0;
                    col[1][i] = c1;
                    col[2][i] = c2;
                    col[3][i] = c3;
                }
            }
        };

        #[cfg(feature = "parallel")]
        bands.par_iter_mut().for_each(process_band);
        #[cfg(not(feature = "parallel"))]
        bands.iter_mut().for_each(process_band);

        self.trace.append(&mut new_cols);
    }

    /// Finalize the trace. Returns the trace and the total sum of the last column.
    /// The last column is shifted by the cumsum_shift.
    pub fn finalize_last(
        mut self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let mut last_col_coords = self.trace.pop().unwrap().columns;

        // Compute cumsum_shift.
        let coordinate_sums = last_col_coords.each_ref().map(|c| {
            c.data
                .iter()
                .copied()
                .sum::<PackedBaseField>()
                .pointwise_sum()
        });
        let claimed_sum = SecureField::from_m31_array(coordinate_sums);
        let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << self.log_size);
        let packed_cumsum_shift = PackedSecureField::broadcast(cumsum_shift);

        last_col_coords.iter_mut().enumerate().for_each(|(i, c)| {
            let shift = packed_cumsum_shift.into_packed_m31s()[i];
            #[cfg(feature = "parallel")]
            c.data.par_iter_mut().for_each(|x| *x -= shift);
            #[cfg(not(feature = "parallel"))]
            c.data.iter_mut().for_each(|x| *x -= shift);
        });

        // The four coordinate prefix sums are independent; run them concurrently.
        #[cfg(feature = "parallel")]
        let coord_prefix_sum = {
            let [c0, c1, c2, c3] = last_col_coords;
            let ((p0, p1), (p2, p3)) = rayon::join(
                || rayon::join(|| inclusive_prefix_sum(c0), || inclusive_prefix_sum(c1)),
                || rayon::join(|| inclusive_prefix_sum(c2), || inclusive_prefix_sum(c3)),
            );
            [p0, p1, p2, p3]
        };
        #[cfg(not(feature = "parallel"))]
        let coord_prefix_sum = last_col_coords.map(inclusive_prefix_sum);
        let secure_prefix_sum = SecureColumnByCoords {
            columns: coord_prefix_sum,
        };
        self.trace.push(secure_prefix_sum);
        let trace = self
            .trace
            .into_iter()
            .flat_map(|eval| {
                eval.columns.map(|col| {
                    CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
                })
            })
            .collect_vec();
        (trace, claimed_sum)
    }
}

/// Trace generator for a single lookup column.
pub struct LogupColGenerator<'a> {
    gen: &'a mut LogupTraceGenerator,
    /// Numerator expressions (i.e. multiplicities) being generated for the current lookup.
    numerator: SecureColumnByCoords<SimdBackend>,
}
impl LogupColGenerator<'_> {
    /// Write a fraction to the column at a row.
    pub fn write_frac(
        &mut self,
        vec_row: usize,
        numerator: PackedSecureField,
        denom: PackedSecureField,
    ) {
        assert_no_logup_poles(denom);
        unsafe {
            self.numerator.set_packed(vec_row, numerator);
            *self.gen.denom.data.get_unchecked_mut(vec_row) = denom;
        }
    }

    /// Finalizes generating the column.
    pub fn finalize_col(mut self) {
        // Column size is a power of 2. The chunk is the rayon task granularity: large
        // enough to amortize scheduling overhead, small enough to balance across threads.
        let chunk_size = std::cmp::min(1 << 10, self.gen.denom.data.len());
        batch_inverse_packed_qm31(&self.gen.denom.data, &mut self.gen.batch_inverse_buffer);

        #[cfg(feature = "parallel")]
        let chunks_iter = {
            let denom_inv_chunks = self.gen.batch_inverse_buffer.par_chunks(chunk_size);
            let numerator_chunks = self.numerator.par_chunks_mut(chunk_size);
            (numerator_chunks, denom_inv_chunks).into_par_iter()
        };

        #[cfg(not(feature = "parallel"))]
        let chunks_iter = {
            let denom_inv_chunks = self.gen.batch_inverse_buffer.chunks(chunk_size);
            let numerator_chunks = self.numerator.chunks_mut(chunk_size);
            numerator_chunks.zip(denom_inv_chunks)
        };

        chunks_iter
            .enumerate()
            .for_each(|(chunk_idx, (mut numerator_chunk, denom_inv_chunk))| {
                for (idx_in_chunk, denom_item) in denom_inv_chunk.iter().enumerate() {
                    unsafe {
                        let vec_row = chunk_idx * chunk_size + idx_in_chunk;
                        let value = numerator_chunk.packed_at(idx_in_chunk) * *denom_item;
                        let prev_value = self
                            .gen
                            .trace
                            .last()
                            .map(|col| col.packed_at(vec_row))
                            .unwrap_or_else(PackedSecureField::zero);
                        numerator_chunk.set_packed(idx_in_chunk, value + prev_value)
                    }
                }
            });

        self.gen.trace.push(self.numerator)
    }

    // TODO(Ohad): remove.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = FractionWriter<'_>> {
        let denom = self.gen.denom.data.iter_mut();
        let [coord0, coord1, coord2, coord3] =
            self.numerator.columns.each_mut().map(|s| &mut s.data);
        multizip((coord0, coord1, coord2, coord3, denom)).map(|(n0, n1, n2, n3, d)| {
            FractionWriter {
                numerator: [n0, n1, n2, n3],
                denom: d,
            }
        })
    }

    // TODO(Ohad): remove.
    #[cfg(feature = "parallel")]
    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = FractionWriter<'_>> {
        let [coord0, coord1, coord2, coord3] =
            self.numerator.columns.each_mut().map(|s| &mut s.data);
        (coord0, coord1, coord2, coord3, &mut self.gen.denom.data)
            .into_par_iter()
            .map(|(n0, n1, n2, n3, d)| FractionWriter {
                numerator: [n0, n1, n2, n3],
                denom: d,
            })
    }
}

/// Exposes a writer for writing a fraction to a single index in a column.
// TODO(Ohad): iterate in chunks, consider VeryPacked.
pub struct FractionWriter<'a> {
    numerator: [&'a mut PackedBaseField; SECURE_EXTENSION_DEGREE],
    denom: &'a mut PackedSecureField,
}
impl FractionWriter<'_> {
    pub fn write_frac(self, numerator: PackedSecureField, denom: PackedSecureField) {
        assert_no_logup_poles(denom);
        let [c0, c1, c2, c3] = numerator.into_packed_m31s();
        *self.numerator[0] = c0;
        *self.numerator[1] = c1;
        *self.numerator[2] = c2;
        *self.numerator[3] = c3;
        *self.denom = denom;
    }
}

#[cfg(test)]
mod tests;
