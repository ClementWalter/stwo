use std::iter::zip;

use itertools::Itertools;
use num_traits::{One, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::CpuBackend;
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
use crate::core::constraints::{coset_vanishing, coset_vanishing_derivative, point_vanishing};
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{batch_inverse_in_place, ExtensionOf};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold, get_folding_alphas};
use crate::core::utils::{bit_reverse, bit_reverse_index};
use crate::prover::backend::{Col, Column};
use crate::prover::fri::FriOps;
use crate::prover::mempool::BaseColumnPool;
use crate::prover::poly::circle::{
    CircleCoefficients, CircleEvaluation, EvalsOrCoeffs, PolyOps, SecureEvaluation,
};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::Poly;

impl PolyOps for CpuBackend {
    type Twiddles = Vec<BaseField>;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        assert!(eval.domain.half_coset.is_doubling_of(twiddles.root_coset));

        // Large transforms dispatch to the shared SIMD FFT (cached packed twiddles, one
        // aligned copy in, zero-copy out); [`interpolate_scalar`] remains the reference.
        if eval.domain.log_size() >= SIMD_DISPATCH_LOG_SIZE {
            use crate::prover::backend::simd::circle::ifft_in_place_raw;
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::prover::backend::simd::SimdBackend;
            let simd_twiddles = cached_simd_twiddles(twiddles.root_coset);
            let domain = eval.domain;
            let mut values = eval.values;

            // Apple-GPU path: in-place transform leaving natural-order coefficients.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if crate::prover::backend::metal::fft::ifft_metal(&mut values, domain, twiddles) {
                return CircleCoefficients::new(values);
            }

            // In-place transform on the CPU-owned allocation when it is SIMD-aligned;
            // no packing copy and no intermediate allocation.
            if let Some(ptr) = simd_aligned_ptr(&mut values) {
                unsafe {
                    ifft_in_place_raw(ptr, domain, &simd_twiddles);
                    convert_simd_coeff_order_raw(ptr, domain.log_size());
                }
                return CircleCoefficients::new(values);
            }

            let packed = BaseColumn::from_cpu(&values);
            let mut coeffs = crate::prover::poly::circle::CircleEvaluation::<
                SimdBackend,
                BaseField,
                BitReversedOrder,
            >::new(domain, packed)
            .interpolate_with_twiddles(&simd_twiddles);
            convert_simd_coeff_order(&mut coeffs.coeffs);
            return CircleCoefficients::new(coeffs.coeffs.into_cpu_vec());
        }

        interpolate_scalar(eval, twiddles)
    }

    fn interpolate_columns(
        columns: Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        // Apple-GPU path: every column's transform in one submission.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let columns = match crate::prover::backend::metal::fft::ifft_batch_metal(columns, twiddles)
        {
            Ok(polys) => return polys,
            Err(columns) => columns,
        };

        #[cfg(feature = "parallel")]
        let iter = columns.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.into_iter();
        iter.map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()
    }

    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        if poly.log_size() == 0 {
            return poly.coeffs[0].into();
        }

        let mut mappings = vec![point.y];
        let mut x = point.x;
        for _ in 1..poly.log_size() {
            mappings.push(x);
            x = CirclePoint::double_x(x);
        }
        mappings.reverse();

        fold(&poly.coeffs, &mappings)
    }

    /// Interpolates and low-degree extends committed columns through the shared SIMD FFT
    /// kernels when the columns are large enough: a column's values are copied once into
    /// an aligned packed buffer, transformed, and handed back without further copies. The
    /// scalar transforms remain the implementation of [`PolyOps::interpolate`] /
    /// [`PolyOps::evaluate`] and a reference test pins both paths equal.
    fn interpolate_and_evaluate_polynomials(
        columns: Vec<EvalsOrCoeffs<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
        store_polynomials_coefficients: bool,
        pool: &BaseColumnPool<Self>,
    ) -> Vec<Poly<Self>> {
        use crate::prover::backend::simd::column::BaseColumn;
        use crate::prover::backend::simd::SimdBackend;
        use crate::prover::poly::circle::CircleEvaluation as GenericCircleEvaluation;

        const MIN_SIMD_DISPATCH_LOG_SIZE: u32 = 10;
        let all_large = columns.iter().all(|column| {
            let log_size = match column {
                EvalsOrCoeffs::Evals(evals) => evals.domain.log_size(),
                EvalsOrCoeffs::Coeffs(coeffs) => coeffs.log_size(),
            };
            log_size >= MIN_SIMD_DISPATCH_LOG_SIZE
        });
        if columns.is_empty() || !all_large {
            // Fall back to the scalar single-pass default.
            return fallback_interpolate_and_evaluate_polynomials(
                columns,
                log_blowup_factor,
                twiddles,
                store_polynomials_coefficients,
                pool,
            );
        }

        // Apple-GPU path: all columns' transforms encoded into one command buffer
        // (single synchronization), natural-order coefficients throughout.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let columns = {
            match crate::prover::backend::metal::fft::fused_transform_metal(
                columns,
                log_blowup_factor,
                twiddles,
                store_polynomials_coefficients,
            ) {
                Ok(polys) => return polys,
                Err(columns) => columns,
            }
        };

        // One cached packed twiddle tree per root coset, shared by all columns of every
        // commitment over that coset.
        let simd_twiddles = cached_simd_twiddles(twiddles.root_coset);

        // Packing fallback for allocations that miss SIMD alignment (rare: large
        // allocations come straight from the page allocator).
        let process_packed = |column: EvalsOrCoeffs<Self>| {
            let mut simd_coeffs = match column {
                EvalsOrCoeffs::Evals(evals) => {
                    let domain = evals.domain;
                    let packed = BaseColumn::from_cpu(&evals.values);
                    GenericCircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                        domain, packed,
                    )
                    .interpolate_with_twiddles(&simd_twiddles)
                }
                EvalsOrCoeffs::Coeffs(coeffs) => {
                    // CPU coefficients are in natural order; the SIMD rfft consumes its
                    // large-transform layout, so convert on the way in.
                    let mut simd_coeffs =
                        crate::prover::poly::circle::CircleCoefficients::<SimdBackend>::new(
                            coeffs.coeffs.into_iter().collect(),
                        );
                    convert_simd_coeff_order(&mut simd_coeffs.coeffs);
                    simd_coeffs
                }
            };
            let ext_domain =
                CanonicCoset::new(simd_coeffs.log_size() + log_blowup_factor).circle_domain();
            let evals = simd_coeffs.evaluate_with_twiddles(ext_domain, &simd_twiddles);
            let evals = CircleEvaluation::<Self, BaseField, BitReversedOrder>::new(
                ext_domain,
                evals.values.into_cpu_vec(),
            );
            let coeffs = store_polynomials_coefficients.then(|| {
                // Stored coefficients feed scalar consumers (OODS sampling, FRI
                // decomposition), which require natural order.
                convert_simd_coeff_order(&mut simd_coeffs.coeffs);
                CircleCoefficients::<Self>::new(simd_coeffs.coeffs.into_cpu_vec())
            });
            Poly::new(coeffs, evals)
        };

        let process = |column: EvalsOrCoeffs<Self>| {
            use crate::prover::backend::simd::circle::{ifft_in_place_raw, rfft_raw};

            // Fast path: both transforms run in place on / straight between the
            // CPU-owned allocations, with no packing copies and no intermediate
            // allocations. The ifft leaves the kernel's coefficient layout in `values`,
            // which is exactly what the rfft consumes; stored coefficients are
            // converted to natural order afterwards for scalar consumers (OODS
            // sampling, FRI decomposition).
            let (mut values, domain, already_coeffs) = match column {
                EvalsOrCoeffs::Evals(evals) => (evals.values, evals.domain, false),
                EvalsOrCoeffs::Coeffs(coeffs) => {
                    let domain = CanonicCoset::new(coeffs.log_size()).circle_domain();
                    (coeffs.coeffs, domain, true)
                }
            };
            let ext_domain =
                CanonicCoset::new(domain.log_size() + log_blowup_factor).circle_domain();

            let mut out: Vec<BaseField> = Vec::with_capacity(ext_domain.size());
            let aligned = simd_aligned_ptr(&mut values)
                .filter(|_| (out.as_mut_ptr() as usize).is_multiple_of(64));
            let Some(src) = aligned else {
                let column = if already_coeffs {
                    EvalsOrCoeffs::Coeffs(CircleCoefficients::new(values))
                } else {
                    EvalsOrCoeffs::Evals(CircleEvaluation::new(domain, values))
                };
                return process_packed(column);
            };
            unsafe {
                if already_coeffs {
                    // Stored natural order -> kernel layout for the rfft.
                    convert_simd_coeff_order_raw(src, domain.log_size());
                } else {
                    ifft_in_place_raw(src, domain, &simd_twiddles);
                }
                rfft_raw(
                    src,
                    out.as_mut_ptr() as *mut u32,
                    domain.log_size(),
                    ext_domain,
                    &simd_twiddles,
                );
                out.set_len(ext_domain.size());
            }
            let coeffs = store_polynomials_coefficients.then(|| {
                unsafe { convert_simd_coeff_order_raw(src, domain.log_size()) };
                CircleCoefficients::<Self>::new(values)
            });
            Poly::new(coeffs, CircleEvaluation::new(ext_domain, out))
        };

        #[cfg(feature = "parallel")]
        return columns.into_par_iter().map(process).collect();
        #[cfg(not(feature = "parallel"))]
        columns.into_iter().map(process).collect()
    }

    fn eval_basis_at_point(log_size: u32, point: CirclePoint<SecureField>) -> Vec<SecureField> {
        if log_size == 0 {
            return vec![SecureField::one()];
        }
        // Folding factors in [`fold`]'s order: factors[0] selects the most significant
        // coefficient-index bit, so the basis doubles through them in reverse.
        let mut mappings = vec![point.y];
        let mut x = point.x;
        for _ in 1..log_size {
            mappings.push(x);
            x = CirclePoint::double_x(x);
        }
        mappings.reverse();

        let mut basis = Vec::with_capacity(1 << log_size);
        basis.push(SecureField::one());
        for &m in mappings.iter().rev() {
            let len = basis.len();
            // The high half is the low half scaled by the factor.
            #[cfg(feature = "parallel")]
            if len >= 1 << 14 {
                let mut high: Vec<SecureField> = Vec::with_capacity(len);
                basis[..len]
                    .par_iter()
                    .map(|&b| b * m)
                    .collect_into_vec(&mut high);
                basis.extend_from_slice(&high);
                continue;
            }
            for i in 0..len {
                basis.push(basis[i] * m);
            }
        }
        basis
    }

    fn eval_at_point_with_basis(
        poly: &CircleCoefficients<Self>,
        basis: &Vec<SecureField>,
    ) -> SecureField {
        assert_eq!(poly.coeffs.len(), basis.len());
        // Four independent partial sums fill the multiplier pipeline; field addition is
        // associative, so the result equals the sequential fold.
        let mut acc = [SecureField::zero(); 4];
        let (coeff_chunks, coeff_rem) = poly.coeffs.as_chunks::<4>();
        let (basis_chunks, basis_rem) = basis.as_chunks::<4>();
        for (cs, bs) in zip(coeff_chunks, basis_chunks) {
            for k in 0..4 {
                acc[k] += bs[k] * cs[k];
            }
        }
        let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
        for (&c, &b) in zip(coeff_rem, basis_rem) {
            sum += b * c;
        }
        sum
    }

    /// Column-blocked shared-basis evaluation: the QM31 basis is deinterleaved once into
    /// its four M31 coordinate columns, and each row band streams the basis through cache
    /// while multiply-accumulating every polynomial's coefficients against it with packed
    /// 16-lane M31 ops. Summation order differs from the sequential fold only by
    /// associativity, so values are exactly equal.
    fn eval_many_at_point_with_basis(
        polys: &[&CircleCoefficients<Self>],
        basis: &Vec<SecureField>,
    ) -> Vec<SecureField> {
        use crate::prover::backend::simd::m31::{PackedBaseField, N_LANES};
        let n = basis.len();
        if polys.len() < 2 || n < (1 << 12) || !n.is_multiple_of(N_LANES) {
            return polys
                .iter()
                .map(|poly| Self::eval_at_point_with_basis(poly, basis))
                .collect();
        }
        polys
            .iter()
            .for_each(|poly| assert_eq!(poly.coeffs.len(), n));

        // Deinterleave the shared basis into coordinate columns (one pass, reused by
        // every polynomial).
        const BAND: usize = 1 << 14;
        let mut coords: [Vec<BaseField>; 4] = std::array::from_fn(|_| vec![BaseField::zero(); n]);
        {
            type FillItem<'a> = (
                (
                    (
                        (&'a mut [BaseField], &'a mut [BaseField]),
                        &'a mut [BaseField],
                    ),
                    &'a mut [BaseField],
                ),
                &'a [SecureField],
            );
            let [c0, c1, c2, c3] = &mut coords;
            let fill = |((((c0, c1), c2), c3), src): FillItem<'_>| {
                for (i, b) in src.iter().enumerate() {
                    [c0[i], c1[i], c2[i], c3[i]] = b.to_m31_array();
                }
            };
            #[cfg(feature = "parallel")]
            c0.par_chunks_mut(BAND)
                .zip(c1.par_chunks_mut(BAND))
                .zip(c2.par_chunks_mut(BAND))
                .zip(c3.par_chunks_mut(BAND))
                .zip(basis.par_chunks(BAND))
                .for_each(fill);
            #[cfg(not(feature = "parallel"))]
            c0.chunks_mut(BAND)
                .zip(c1.chunks_mut(BAND))
                .zip(c2.chunks_mut(BAND))
                .zip(c3.chunks_mut(BAND))
                .zip(basis.chunks(BAND))
                .for_each(fill);
        }
        let bands: Vec<usize> = (0..n).step_by(BAND).collect();
        let accumulate_band = |&start: &usize| {
            let end = (start + BAND).min(n);
            // Polynomial-outer: each polynomial's band streams contiguously against the
            // cache-resident basis band, with the four coordinate accumulators held in
            // registers.
            polys
                .iter()
                .map(|poly| {
                    let mut acc = [PackedBaseField::zero(); 4];
                    let mut idx = start;
                    while idx < end {
                        let c = PackedBaseField::from_array(
                            poly.coeffs[idx..idx + N_LANES].try_into().unwrap(),
                        );
                        for k in 0..4 {
                            let b = PackedBaseField::from_array(
                                coords[k][idx..idx + N_LANES].try_into().unwrap(),
                            );
                            acc[k] += b * c;
                        }
                        idx += N_LANES;
                    }
                    acc
                })
                .collect_vec()
        };

        #[cfg(feature = "parallel")]
        let partials: Vec<Vec<[PackedBaseField; 4]>> =
            bands.par_iter().map(accumulate_band).collect();
        #[cfg(not(feature = "parallel"))]
        let partials: Vec<Vec<[PackedBaseField; 4]>> = bands.iter().map(accumulate_band).collect();

        (0..polys.len())
            .map(|p| {
                let coords: [BaseField; 4] = std::array::from_fn(|k| {
                    partials
                        .iter()
                        .map(|band| band[p][k])
                        .fold(PackedBaseField::zero(), |a, b| a + b)
                        .pointwise_sum()
                });
                SecureField::from_m31_array(coords)
            })
            .collect()
    }

    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<CpuBackend, SecureField> {
        let domain = coset.circle_domain();

        let (si_i, vi_p): (Vec<_>, Vec<_>) = (0..domain.size())
            .map(|i| {
                let coset_point = domain
                    .at(bit_reverse_index(i, domain.log_size()))
                    .into_ef::<SecureField>();
                let minus_two_coset_point_y = coset_point.y * SecureField::from(-2);
                (
                    minus_two_coset_point_y
                        * coset_vanishing_derivative(
                            Coset::new(CirclePointIndex::generator(), domain.log_size()),
                            coset_point,
                        ),
                    point_vanishing(coset_point, p.into_ef::<SecureField>()),
                )
            })
            .unzip();

        let vn_p: SecureField = coset_vanishing(
            CanonicCoset::new(domain.log_size()).coset,
            p.into_ef::<SecureField>(),
        );

        (0..domain.size())
            .map(|i| vn_p / (si_i[i] * vi_p[i]))
            .collect_vec()
    }

    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>,
        weights: &Col<CpuBackend, SecureField>,
    ) -> SecureField {
        (0..evals.domain.size()).fold(SecureField::zero(), |acc, i| {
            acc + (evals.values[i] * weights[i])
        })
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        let log_size = evals.domain.log_size();
        let mut folding_alphas = get_folding_alphas(point, log_size as usize);

        let secure_field_values: Vec<SecureField> = evals
            .values
            .to_cpu()
            .iter()
            .map(|f| SecureField::from(*f))
            .collect_vec();

        let mut layer_evaluation = CpuBackend::fold_circle_into_line(
            &SecureEvaluation::new(
                evals.domain,
                SecureColumnByCoords::from_iter(secure_field_values),
            ),
            folding_alphas.pop().unwrap(),
            twiddles,
        );

        while layer_evaluation.len() > 1 {
            let alpha = folding_alphas.pop().unwrap();
            layer_evaluation = CpuBackend::fold_line(&layer_evaluation, &[alpha], twiddles);
        }

        layer_evaluation.values.at(0) / SecureField::from(2_u32.pow(log_size))
    }

    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        assert!(log_size >= poly.log_size());
        let mut coeffs = Vec::with_capacity(1 << log_size);
        coeffs.extend_from_slice(&poly.coeffs);
        coeffs.resize(1 << log_size, BaseField::zero());
        CircleCoefficients::new(coeffs)
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // The SIMD-dispatched path of evaluate_into allocates its own aligned buffer;
        // only allocate the scalar buffer when the scalar path will run.
        if domain.log_size() >= SIMD_DISPATCH_LOG_SIZE
            && poly.coeffs.len().ilog2() >= SIMD_DISPATCH_LOG_SIZE
        {
            return Self::evaluate_into(poly, domain, twiddles, Vec::new());
        }
        let buffer = vec![BaseField::zero(); domain.size()];
        Self::evaluate_into(poly, domain, twiddles, buffer)
    }

    fn evaluate_into(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
        buffer: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        assert!(domain.half_coset.is_doubling_of(twiddles.root_coset));

        // Large transforms dispatch to the shared SIMD FFT; see [`PolyOps::interpolate`].
        // The polynomial itself must also clear the dispatch size: the SIMD backend's
        // small-input fallback is this very function. The dispatched path allocates its
        // own aligned output, ignoring `buffer`.
        if domain.log_size() >= SIMD_DISPATCH_LOG_SIZE
            && poly.coeffs.len().ilog2() >= SIMD_DISPATCH_LOG_SIZE
        {
            use crate::prover::backend::simd::circle::rfft_raw;
            use crate::prover::backend::simd::fft::CACHED_FFT_LOG_SIZE;
            use crate::prover::backend::simd::m31::N_LANES;
            use crate::prover::backend::simd::SimdBackend;
            let simd_twiddles = cached_simd_twiddles(twiddles.root_coset);
            let fft_log_size = poly.coeffs.len().ilog2();

            // Apple-GPU path: transform straight from the stored natural-order
            // coefficients into a fresh output vector.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                let mut out = vec![BaseField::zero(); domain.size()];
                if crate::prover::backend::metal::fft::rfft_metal(
                    &poly.coeffs,
                    domain,
                    twiddles,
                    &mut out,
                ) {
                    return CircleEvaluation::new(domain, out);
                }
            }

            // Raw path: run the rfft straight from the CPU coefficient buffer into a
            // fresh output vector — no packing copies. Above the cached-fft size the
            // kernel consumes its transposed layout, so a scratch copy is converted
            // first; at or below it the layouts coincide and the stored coefficients
            // are read directly.
            let mut out: Vec<BaseField> = Vec::with_capacity(domain.size());
            if (out.as_mut_ptr() as usize).is_multiple_of(64) {
                let mut scratch: Vec<BaseField> = Vec::new();
                let src: Option<*const u32> = if fft_log_size > CACHED_FFT_LOG_SIZE {
                    scratch = poly.coeffs.clone();
                    simd_aligned_ptr(&mut scratch).map(|ptr| {
                        unsafe { convert_simd_coeff_order_raw(ptr, fft_log_size) };
                        ptr as *const u32
                    })
                } else {
                    let ptr = poly.coeffs.as_ptr() as usize;
                    (ptr.is_multiple_of(64) && poly.coeffs.len().is_multiple_of(N_LANES))
                        .then_some(ptr as *const u32)
                };
                if let Some(src) = src {
                    unsafe {
                        rfft_raw(
                            src,
                            out.as_mut_ptr() as *mut u32,
                            fft_log_size,
                            domain,
                            &simd_twiddles,
                        );
                        out.set_len(domain.size());
                    }
                    drop(scratch);
                    return CircleEvaluation::new(domain, out);
                }
            }

            let mut simd_coeffs =
                crate::prover::poly::circle::CircleCoefficients::<SimdBackend>::new(
                    poly.coeffs.iter().copied().collect(),
                );
            convert_simd_coeff_order(&mut simd_coeffs.coeffs);
            let evals = simd_coeffs.evaluate_with_twiddles(domain, &simd_twiddles);
            return CircleEvaluation::new(domain, evals.values.into_cpu_vec());
        }

        assert_eq!(buffer.len(), domain.size());
        evaluate_into_scalar(poly, domain, twiddles, buffer)
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // Apple-GPU path; bit-identical values (unique inverses, exact group ops).
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(tree) =
            crate::prover::backend::metal::twiddles::precompute_twiddles_metal(coset)
        {
            return tree;
        }

        const CHUNK_LOG_SIZE: usize = 12;
        const CHUNK_SIZE: usize = 1 << CHUNK_LOG_SIZE;

        let root_coset = coset;
        let twiddles = slow_precompute_twiddles(coset);

        // Inverse twiddles.
        // Fallback to the non-chunked version if the domain is not big enough.
        if CHUNK_SIZE > root_coset.size() {
            let itwiddles = twiddles.iter().map(|&t| t.inverse()).collect();
            return TwiddleTree {
                root_coset,
                twiddles,
                itwiddles,
            };
        }

        let mut itwiddles = vec![BaseField::zero(); twiddles.len()];
        #[cfg(feature = "parallel")]
        twiddles
            .as_chunks::<CHUNK_SIZE>()
            .0
            .par_iter()
            .zip(itwiddles.as_chunks_mut::<CHUNK_SIZE>().0.par_iter_mut())
            .for_each(|(src, dst)| {
                batch_inverse_in_place(src, dst);
            });
        #[cfg(not(feature = "parallel"))]
        twiddles
            .array_chunks::<CHUNK_SIZE>()
            .zip(itwiddles.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|(src, dst)| {
                batch_inverse_in_place(src, dst);
            });

        TwiddleTree {
            root_coset,
            twiddles,
            itwiddles,
        }
    }

    fn split_at_mid(
        mut poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let right = poly.coeffs.split_off(poly.coeffs.len() / 2);
        (
            CircleCoefficients::new(poly.coeffs),
            CircleCoefficients::new(right),
        )
    }
}

/// Returns a cached SIMD twiddle tree for `root_coset`, building it on first use. The
/// proof's transforms reuse one or two distinct root cosets, so the cache stays tiny.
#[allow(clippy::type_complexity)]
pub(crate) fn cached_simd_twiddles(
    root_coset: Coset,
) -> std::sync::Arc<TwiddleTree<crate::prover::backend::simd::SimdBackend>> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use crate::prover::backend::simd::SimdBackend;

    static CACHE: OnceLock<Mutex<HashMap<(u32, u32), Arc<TwiddleTree<SimdBackend>>>>> =
        OnceLock::new();
    let key = (root_coset.initial_index.0 as u32, root_coset.log_size);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(tree) = cache.lock().unwrap().get(&key) {
        return tree.clone();
    }
    let tree = Arc::new(SimdBackend::precompute_twiddles(root_coset));
    cache.lock().unwrap().entry(key).or_insert(tree).clone()
}

/// Size from which single-column transforms dispatch to the shared SIMD FFT kernels.
const SIMD_DISPATCH_LOG_SIZE: u32 = 10;

/// Converts a coefficient column between the SIMD kernels' large-transform layout and
/// natural coefficient order. Above [`CACHED_FFT_LOG_SIZE`] the SIMD ifft leaves
/// coefficients vec-transposed (its rfft and eval_at_point consume that same layout),
/// while every scalar consumer requires natural order — so coefficient columns crossing
/// the CPU<->SIMD dispatch boundary must be converted. The transpose is an involution,
/// hence one function serves both directions. Below the threshold the layouts coincide
/// and this is a no-op.
pub(crate) fn convert_simd_coeff_order(
    column: &mut crate::prover::backend::simd::column::BaseColumn,
) {
    // Safe: PackedBaseField data is 64-byte aligned and fully initialized.
    unsafe {
        convert_simd_coeff_order_raw(column.data.as_mut_ptr() as *mut u32, column.len().ilog2());
    }
}

/// Raw-pointer form of [`convert_simd_coeff_order`] for coefficient buffers held in
/// CPU-owned allocations.
///
/// # Safety
///
/// `ptr` must be 64-byte aligned and valid for reads and writes of `2^log_size` u32s.
pub(crate) unsafe fn convert_simd_coeff_order_raw(ptr: *mut u32, log_size: u32) {
    use crate::prover::backend::simd::fft::{transpose_vecs, CACHED_FFT_LOG_SIZE};
    use crate::prover::backend::simd::m31::LOG_N_LANES;
    if log_size > CACHED_FFT_LOG_SIZE {
        transpose_vecs(ptr, (log_size - LOG_N_LANES) as usize);
    }
}

/// Returns the column's base pointer when the allocation happens to satisfy the SIMD
/// kernels' 64-byte alignment, allowing in-place transforms without packing copies.
/// Large allocations come straight from the page allocator, so this holds in practice;
/// callers must keep a packing fallback for when it doesn't.
fn simd_aligned_ptr(values: &mut [BaseField]) -> Option<*mut u32> {
    use crate::prover::backend::simd::m31::N_LANES;
    let ptr = values.as_mut_ptr() as usize;
    (ptr.is_multiple_of(64) && values.len().is_multiple_of(N_LANES)).then_some(ptr as *mut u32)
}

/// Scalar circle FFT into a caller-provided buffer, the reference implementation behind
/// [`PolyOps::evaluate_into`]'s SIMD dispatch.
pub(crate) fn evaluate_into_scalar(
    poly: &CircleCoefficients<CpuBackend>,
    domain: CircleDomain,
    twiddles: &TwiddleTree<CpuBackend>,
    mut buffer: Col<CpuBackend, BaseField>,
) -> CircleEvaluation<CpuBackend, BaseField, BitReversedOrder> {
    // Copy extended coefficients into the buffer.
    let poly_len = poly.coeffs.len();
    buffer[..poly_len].copy_from_slice(&poly.coeffs);
    for v in &mut buffer[poly_len..] {
        *v = BaseField::zero();
    }

    if domain.log_size() == 1 {
        let (mut v0, mut v1) = (buffer[0], buffer[1]);
        butterfly(&mut v0, &mut v1, domain.half_coset.initial.y);
        buffer[0] = v0;
        buffer[1] = v1;
        return CircleEvaluation::new(domain, buffer);
    }

    if domain.log_size() == 2 {
        let (mut v0, mut v1, mut v2, mut v3) = (buffer[0], buffer[1], buffer[2], buffer[3]);
        let CirclePoint { x, y } = domain.half_coset.initial;
        butterfly(&mut v0, &mut v2, x);
        butterfly(&mut v1, &mut v3, x);
        butterfly(&mut v0, &mut v1, y);
        butterfly(&mut v2, &mut v3, -y);
        buffer[0] = v0;
        buffer[1] = v1;
        buffer[2] = v2;
        buffer[3] = v3;
        return CircleEvaluation::new(domain, buffer);
    }

    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);
    let circle_twiddles = circle_twiddles_from_line_twiddles(line_twiddles[0]).collect_vec();

    for (layer, layer_twiddles) in line_twiddles.iter().enumerate().rev() {
        fft_full_layer(&mut buffer, layer + 1, |h| layer_twiddles[h], butterfly);
    }
    fft_full_layer(&mut buffer, 0, |h| circle_twiddles[h], butterfly);

    CircleEvaluation::new(domain, buffer)
}

/// Scalar circle iFFT, the reference implementation behind
/// [`PolyOps::interpolate`]'s SIMD dispatch.
pub(crate) fn interpolate_scalar(
    eval: CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<CpuBackend>,
) -> CircleCoefficients<CpuBackend> {
    let mut values = eval.values;

    if eval.domain.log_size() == 1 {
        let y = eval.domain.half_coset.initial.y;
        let n = BaseField::from(2);
        let yn_inv = (y * n).inverse();
        let y_inv = yn_inv * n;
        let n_inv = yn_inv * y;
        let (mut v0, mut v1) = (values[0], values[1]);
        ibutterfly(&mut v0, &mut v1, y_inv);
        return CircleCoefficients::new(vec![v0 * n_inv, v1 * n_inv]);
    }

    if eval.domain.log_size() == 2 {
        let CirclePoint { x, y } = eval.domain.half_coset.initial;
        let n = BaseField::from(4);
        let xyn_inv = (x * y * n).inverse();
        let x_inv = xyn_inv * y * n;
        let y_inv = xyn_inv * x * n;
        let n_inv = xyn_inv * x * y;
        let (mut v0, mut v1, mut v2, mut v3) = (values[0], values[1], values[2], values[3]);
        ibutterfly(&mut v0, &mut v1, y_inv);
        ibutterfly(&mut v2, &mut v3, -y_inv);
        ibutterfly(&mut v0, &mut v2, x_inv);
        ibutterfly(&mut v1, &mut v3, x_inv);
        return CircleCoefficients::new(vec![v0 * n_inv, v1 * n_inv, v2 * n_inv, v3 * n_inv]);
    }

    let line_twiddles = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);
    let circle_twiddles = circle_twiddles_from_line_twiddles(line_twiddles[0]).collect_vec();

    fft_full_layer(&mut values, 0, |h| circle_twiddles[h], ibutterfly);
    for (layer, layer_twiddles) in line_twiddles.into_iter().enumerate() {
        fft_full_layer(&mut values, layer + 1, |h| layer_twiddles[h], ibutterfly);
    }

    // Divide all values by 2^log_size.
    let inv = BaseField::from_u32_unchecked(eval.domain.size() as u32).inverse();
    #[cfg(feature = "parallel")]
    values.par_iter_mut().for_each(|val| *val *= inv);
    #[cfg(not(feature = "parallel"))]
    for val in &mut values {
        *val *= inv;
    }

    CircleCoefficients::new(values)
}

pub fn slow_precompute_twiddles(mut coset: Coset) -> Vec<BaseField> {
    let mut twiddles = Vec::with_capacity(coset.size());
    for _ in 0..coset.log_size() {
        let i0 = twiddles.len();
        let half = coset.size() / 2;
        // Each chunk derives its starting point by index and steps from there, so the
        // layer's points are computed in parallel.
        const CHUNK: usize = 1 << 12;
        let mut layer = vec![BaseField::zero(); half];
        let fill = |(chunk_idx, chunk): (usize, &mut [BaseField])| {
            let mut point = coset.at(chunk_idx * CHUNK);
            for slot in chunk.iter_mut() {
                *slot = point.x;
                point = point + coset.step;
            }
        };
        #[cfg(feature = "parallel")]
        layer
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| fill((i, chunk)));
        #[cfg(not(feature = "parallel"))]
        layer
            .chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| fill((i, chunk)));
        twiddles.extend(layer);
        bit_reverse(&mut twiddles[i0..]);
        coset = coset.double();
    }
    // Pad with an arbitrary value to make the length a power of 2.
    twiddles.push(1.into());
    twiddles
}

/// Applies one FFT layer over the whole value buffer: for each `h`, the butterflies of
/// block `h` (a contiguous range of `2^(i+1)` values, pairing index `l` with
/// `l + 2^i`) use twiddle `twiddle_at(h)`. Blocks are disjoint and butterflies within a
/// block are independent, so the layer is processed in parallel: across blocks when
/// there are many, and across the butterfly pairs of the (few, large) blocks otherwise.
fn fft_full_layer(
    values: &mut [BaseField],
    i: usize,
    twiddle_at: impl Fn(usize) -> BaseField + Sync,
    butterfly_fn: impl Fn(&mut BaseField, &mut BaseField, BaseField) + Sync,
) {
    let block = 1 << (i + 1);
    let half = 1 << i;

    #[cfg(feature = "parallel")]
    {
        const MIN_PAR_SIZE: usize = 1 << 13;
        // Only go parallel from a top-level call: when the FFT is already running inside
        // a rayon task (e.g. one column among many in a per-column parallel pass), inner
        // splitting just adds scheduling overhead.
        if rayon::current_thread_index().is_none() && values.len() >= MIN_PAR_SIZE {
            values
                .par_chunks_mut(block)
                .enumerate()
                .for_each(|(h, chunk)| {
                    let t = twiddle_at(h);
                    let (lo, hi) = chunk.split_at_mut(half);
                    // Within a large block, split the butterfly pairs across threads too.
                    const SUB: usize = 1 << 12;
                    lo.par_chunks_mut(SUB).zip(hi.par_chunks_mut(SUB)).for_each(
                        |(lo_chunk, hi_chunk)| {
                            for (v0, v1) in lo_chunk.iter_mut().zip(hi_chunk.iter_mut()) {
                                butterfly_fn(v0, v1, t);
                            }
                        },
                    );
                });
            return;
        }
    }

    for (h, chunk) in values.chunks_mut(block).enumerate() {
        let t = twiddle_at(h);
        let (lo, hi) = chunk.split_at_mut(half);
        for (v0, v1) in lo.iter_mut().zip(hi.iter_mut()) {
            butterfly_fn(v0, v1, t);
        }
    }
}

/// Computes the circle twiddles layer (layer 0) from the first line twiddles layer (layer 1).
///
/// Only works for line twiddles generated from a domain with size `>4`.
pub(crate) fn circle_twiddles_from_line_twiddles(
    first_line_twiddles: &[BaseField],
) -> impl Iterator<Item = BaseField> + '_ {
    // The twiddles for layer 0 can be computed from the twiddles for layer 1.
    // Since the twiddles are bit reversed, we consider the circle domain in bit reversed order.
    // Each consecutive 4 points in the bit reversed order of a coset form a circle coset of size 4.
    // A circle coset of size 4 in bit reversed order looks like this:
    //   [(x, y), (-x, -y), (y, -x), (-y, x)]
    // Note: This relation is derived from the fact that `M31_CIRCLE_GEN`.repeated_double(ORDER / 4)
    //   == (-1,0), and not (0,1). (0,1) would yield another relation.
    // The twiddles for layer 0 are the y coordinates:
    //   [y, -y, -x, x]
    // The twiddles for layer 1 in bit reversed order are the x coordinates of the even indices
    // points:
    //   [x, y]
    // Works also for inverse of the twiddles.
    first_line_twiddles
        .iter()
        .array_chunks()
        .flat_map(|[&x, &y]| [y, -y, -x, x])
}

impl<F: ExtensionOf<BaseField>, EvalOrder> IntoIterator
    for CircleEvaluation<CpuBackend, F, EvalOrder>
{
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;

    /// Creates a consuming iterator over the evaluations.
    ///
    /// Evaluations are returned in the same order as elements of the domain.
    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::One;

    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::CpuBackend;
    use crate::prover::poly::circle::{CircleEvaluation, PolyOps};
    use crate::prover::poly::BitReversedOrder;

    #[test]
    fn test_eval_at_point_with_4_coeffs() {
        // Represents the polynomial `1 + 2y + 3x + 4xy`.
        // Note coefficients are passed in bit reversed order.
        let poly = CpuCirclePoly::new([1, 3, 2, 4].map(BaseField::from).to_vec());
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(
            eval,
            poly.coeffs[0] + poly.coeffs[1] * y + poly.coeffs[2] * x + poly.coeffs[3] * x * y
        );
    }

    #[test]
    fn test_eval_at_point_with_2_coeffs() {
        // Represents the polynomial `1 + 2y`.
        let poly = CpuCirclePoly::new(vec![BaseField::from(1), BaseField::from(2)]);
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(eval, poly.coeffs[0] + poly.coeffs[1] * y);
    }

    #[test]
    fn test_eval_at_point_with_1_coeff() {
        // Represents the polynomial `1`.
        let poly = CpuCirclePoly::new(vec![BaseField::one()]);
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(eval, SecureField::one());
    }

    #[test]
    fn test_cpu_eval_at_point_by_folding() {
        let poly = CpuCirclePoly::new(
            [691, 805673, 5, 435684, 4832, 23876431, 197, 897346068]
                .map(BaseField::from)
                .to_vec(),
        );
        let s = CanonicCoset::new(10);
        let domain = s.circle_domain();
        let twiddles =
            CpuBackend::precompute_twiddles(CanonicCoset::new(11).circle_domain().half_coset);
        let eval = poly.evaluate(domain);
        let sampled_points = [
            CirclePoint::get_point(348),
            CirclePoint::get_point(9736524),
            CirclePoint::get_point(13),
            CirclePoint::get_point(346752),
        ];
        let sampled_values = sampled_points
            .iter()
            .map(|point| poly.eval_at_point(*point))
            .collect_vec();

        let sampled_folding_values = sampled_points
            .iter()
            .map(|point| eval.eval_at_point_by_folding(*point, &twiddles))
            .collect_vec();

        assert_eq!(
            sampled_folding_values, sampled_values,
            "Evaluation by folding should be equal to the polynomial evaluation"
        );
    }

    #[test]
    fn test_evaluate_2_coeffs() {
        let domain = CanonicCoset::new(1).circle_domain();
        let poly = CpuCirclePoly::new((1..=2).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (p, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(p.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_evaluate_4_coeffs() {
        let domain = CanonicCoset::new(2).circle_domain();
        let poly = CpuCirclePoly::new((1..=4).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_evaluate_8_coeffs() {
        let domain = CanonicCoset::new(3).circle_domain();
        let poly = CpuCirclePoly::new((1..=8).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_interpolate_2_evals() {
        let poly = CpuCirclePoly::new(vec![BaseField::one(), BaseField::from(2)]);
        let domain = CanonicCoset::new(1).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_interpolate_4_evals() {
        let poly = CpuCirclePoly::new((1..=4).map(BaseField::from).collect());
        let domain = CanonicCoset::new(2).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_interpolate_8_evals() {
        let poly = CpuCirclePoly::new((1..=8).map(BaseField::from).collect());
        let domain = CanonicCoset::new(3).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_circle_poly_split_at_mid() {
        let log_size = 4;
        let poly = CpuCirclePoly::new((0..1 << log_size).map(BaseField::from).collect());
        let (left, right) = poly.clone().split_at_mid();
        let random_point = CirclePoint::get_point(21903);

        assert_eq!(
            left.eval_at_point(random_point)
                + random_point.repeated_double(log_size - 2).x * right.eval_at_point(random_point),
            poly.eval_at_point(random_point)
        );
    }

    #[test]
    fn test_cpu_barycentric_evaluation() {
        let poly = CpuCirclePoly::new(
            [691, 805673, 5, 435684, 4832, 23876431, 197, 897346068]
                .map(BaseField::from)
                .to_vec(),
        );
        let s = CanonicCoset::new(10);
        let domain = s.circle_domain();
        let eval = poly.evaluate(domain);
        let sampled_points = [
            CirclePoint::get_point(348),
            CirclePoint::get_point(9736524),
            CirclePoint::get_point(13),
            CirclePoint::get_point(346752),
        ];
        let sampled_values = sampled_points
            .iter()
            .map(|point| poly.eval_at_point(*point))
            .collect_vec();

        let sampled_barycentric_values = sampled_points
            .iter()
            .map(|point| {
                eval.barycentric_eval_at_point(&CircleEvaluation::<
                    CpuBackend,
                    BaseField,
                    BitReversedOrder,
                >::barycentric_weights(s, *point))
            })
            .collect_vec();

        assert_eq!(
            sampled_barycentric_values, sampled_values,
            "Barycentric evaluation should be equal to the polynomial evaluation"
        );
    }
}

/// Scalar single-pass interpolation + extension, the default behavior of
/// [`PolyOps::interpolate_and_evaluate_polynomials`]; used for small columns and as the
/// reference for the SIMD-dispatched path.
fn fallback_interpolate_and_evaluate_polynomials(
    columns: Vec<EvalsOrCoeffs<CpuBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CpuBackend>,
    store_polynomials_coefficients: bool,
    pool: &BaseColumnPool<CpuBackend>,
) -> Vec<Poly<CpuBackend>> {
    let buffers: Vec<_> = columns
        .iter()
        .map(|column| {
            let log_eval_size = match column {
                EvalsOrCoeffs::Evals(evals) => evals.domain.log_size(),
                EvalsOrCoeffs::Coeffs(coeffs) => coeffs.log_size(),
            } + log_blowup_factor;
            pool.take_or_alloc(log_eval_size)
        })
        .collect();

    #[cfg(feature = "parallel")]
    let iter = columns.into_par_iter().zip(buffers.into_par_iter());
    #[cfg(not(feature = "parallel"))]
    let iter = columns.into_iter().zip(buffers);

    iter.map(|(column, buffer)| {
        let poly_coeffs = match column {
            EvalsOrCoeffs::Evals(evals) => evals.interpolate_with_twiddles(twiddles),
            EvalsOrCoeffs::Coeffs(coeffs) => coeffs,
        };
        let domain = CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain();
        let evals = <CpuBackend as PolyOps>::evaluate_into(&poly_coeffs, domain, twiddles, buffer);
        Poly::new(store_polynomials_coefficients.then_some(poly_coeffs), evals)
    })
    .collect()
}

#[cfg(test)]
mod dispatch_tests {
    use itertools::Itertools;

    use super::*;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse_index;
    use crate::prover::poly::circle::{CircleEvaluation, EvalsOrCoeffs, PolyOps};
    use crate::prover::poly::BitReversedOrder;

    fn test_values(log_size: u32) -> Vec<BaseField> {
        (0..1u32 << log_size)
            .map(|i| BaseField::from(i.wrapping_mul(2654435761) >> 4))
            .collect()
    }

    /// The per-layer twiddle slices served to a domain must be independent of the tree's
    /// root size — the invariant behind sharing one big tree across all domain sizes.
    #[test]
    fn twiddle_layers_independent_of_tree_root() {
        let domain = CanonicCoset::new(17).circle_domain();
        let t17 = CpuBackend::precompute_twiddles(CanonicCoset::new(17).circle_domain().half_coset);
        let t18 = CpuBackend::precompute_twiddles(CanonicCoset::new(18).circle_domain().half_coset);
        let l17 = domain_line_twiddles_from_tree(domain, &t17.itwiddles);
        let l18 = domain_line_twiddles_from_tree(domain, &t18.itwiddles);
        assert_eq!(l17.len(), l18.len());
        for (i, (a, b)) in l17.iter().zip(l18.iter()).enumerate() {
            assert_eq!(
                a, b,
                "itwiddle layer {i} differs between root 17 and root 18"
            );
        }
    }

    /// Same root-independence invariant for the SIMD twiddle tree's flat buffer.
    #[test]
    fn simd_twiddle_layers_independent_of_tree_root() {
        use crate::prover::backend::simd::SimdBackend;
        let domain = CanonicCoset::new(17).circle_domain();
        let t17 =
            SimdBackend::precompute_twiddles(CanonicCoset::new(17).circle_domain().half_coset);
        let t18 =
            SimdBackend::precompute_twiddles(CanonicCoset::new(18).circle_domain().half_coset);
        let l17 = domain_line_twiddles_from_tree(domain, &t17.itwiddles);
        let l18 = domain_line_twiddles_from_tree(domain, &t18.itwiddles);
        assert_eq!(l17.len(), l18.len());
        for (i, (a, b)) in l17.iter().zip(l18.iter()).enumerate() {
            assert_eq!(
                a,
                b,
                "simd itwiddle layer {i} (len {}) differs between root 17 and root 18",
                a.len()
            );
        }
    }

    /// Dispatched interpolate vs the scalar reference, spanning the SIMD cached-fft
    /// boundary (2^16) where the SIMD kernel switches to its vec-transposed layout, with
    /// both exact-size and oversized twiddle trees.
    #[test]
    fn dispatched_interpolate_matches_scalar() {
        for (log_size, tree_log) in [
            (10u32, 11u32),
            (15, 18),
            (16, 18),
            (17, 17),
            (17, 18),
            (18, 18),
            (18, 19),
        ] {
            let twiddles = CpuBackend::precompute_twiddles(
                CanonicCoset::new(tree_log).circle_domain().half_coset,
            );
            let domain = CanonicCoset::new(log_size).circle_domain();
            let values = test_values(log_size);
            let eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                domain,
                values.clone(),
            );
            let dispatched = <CpuBackend as PolyOps>::interpolate(eval, &twiddles);
            let eval2 =
                CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(domain, values);
            let scalar = interpolate_scalar(eval2, &twiddles);
            assert_eq!(
                dispatched.coeffs, scalar.coeffs,
                "log_size {log_size} tree {tree_log}"
            );
        }
    }

    /// Dispatched evaluate vs the scalar reference across the cached-fft boundary,
    /// including blown-up domains (poly smaller than the evaluation domain).
    #[test]
    fn dispatched_evaluate_matches_scalar() {
        for (log_size, log_blowup, tree_log) in [
            (10u32, 1u32, 11u32),
            (15, 1, 18),
            (16, 1, 18),
            (17, 1, 18),
            (17, 0, 18),
            (18, 1, 19),
        ] {
            let twiddles = CpuBackend::precompute_twiddles(
                CanonicCoset::new(tree_log).circle_domain().half_coset,
            );
            let poly = CircleCoefficients::<CpuBackend>::new(test_values(log_size));
            let domain = CanonicCoset::new(log_size + log_blowup).circle_domain();
            let dispatched = <CpuBackend as PolyOps>::evaluate(&poly, domain, &twiddles);
            let scalar = evaluate_into_scalar(
                &poly,
                domain,
                &twiddles,
                vec![BaseField::zero(); domain.size()],
            );
            assert_eq!(
                dispatched.values, scalar.values,
                "log_size {log_size} blowup {log_blowup} tree {tree_log}"
            );
        }
    }

    /// FFT-free mathematical oracle, independent of the scalar reference: interpolated
    /// coefficients must evaluate back to the original values at domain points. Catches
    /// coefficient-order corruption that any same-order roundtrip would mask.
    #[test]
    fn dispatched_interpolate_evaluates_back_to_values() {
        let log_size = 17u32;
        let twiddles =
            CpuBackend::precompute_twiddles(CanonicCoset::new(18).circle_domain().half_coset);
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = test_values(log_size);
        let eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            domain,
            values.clone(),
        );
        let coeffs = <CpuBackend as PolyOps>::interpolate(eval, &twiddles);
        for idx in [0usize, 1, 12345, 99999, (1 << log_size) - 1] {
            let point = domain.at(bit_reverse_index(idx, log_size)).into_ef();
            let value = <CpuBackend as PolyOps>::eval_at_point(&coeffs, point);
            assert_eq!(value, values[idx].into(), "domain point {idx}");
        }
    }

    /// The SIMD-dispatched fused interpolation+extension must produce exactly the
    /// explicitly-scalar pipeline's coefficients and evaluations, across both the
    /// dispatch threshold and the SIMD cached-fft boundary.
    #[test]
    fn simd_dispatched_commit_matches_scalar() {
        for log_size in [5u32, 9, 10, 12, 15, 16, 17, 18] {
            let twiddles = CpuBackend::precompute_twiddles(
                CanonicCoset::new(log_size + 1).circle_domain().half_coset,
            );
            let domain = CanonicCoset::new(log_size).circle_domain();
            let make_values = |c: usize| {
                (0..1 << log_size)
                    .map(|i| BaseField::from(((i + 1) * (c + 7)) as u32))
                    .collect_vec()
            };
            let columns = (0..3usize)
                .map(|c| {
                    EvalsOrCoeffs::Evals(
                        CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                            domain,
                            make_values(c),
                        ),
                    )
                })
                .collect_vec();
            let pool = BaseColumnPool::new();
            let dispatched = <CpuBackend as PolyOps>::interpolate_and_evaluate_polynomials(
                columns, 1, &twiddles, true, &pool,
            );
            // The reference side goes through the explicitly-scalar kernels, never the
            // dispatched PolyOps entry points (those are the code under test).
            let ext_domain = CanonicCoset::new(log_size + 1).circle_domain();
            for (c, d) in dispatched.iter().enumerate() {
                let eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                    domain,
                    make_values(c),
                );
                let scalar_coeffs = interpolate_scalar(eval, &twiddles);
                let scalar_evals = evaluate_into_scalar(
                    &scalar_coeffs,
                    ext_domain,
                    &twiddles,
                    vec![BaseField::zero(); ext_domain.size()],
                );
                assert_eq!(
                    d.evals.values, scalar_evals.values,
                    "evals log_size {log_size} col {c}"
                );
                assert_eq!(
                    d.coeffs.as_ref().unwrap().coeffs,
                    scalar_coeffs.coeffs,
                    "coeffs log_size {log_size} col {c}"
                );
            }
        }
    }
}
