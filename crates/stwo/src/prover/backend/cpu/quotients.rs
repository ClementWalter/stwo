use std::iter::zip;

use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::quotients::{denominators, quotient_constants, ColumnSampleBatch};
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::twiddles::{TwiddleBuffer, TwiddleTree};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

impl QuotientOps for CpuBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    ) {
        let size = columns[0].len();
        let subdomain_size = size >> log_blowup_factor;
        let quotient_constants = quotient_constants(sample_batches);

        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            // Apple-GPU path: the whole batch accumulated in one GPU submission.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if subdomain_size
                >= 1 << crate::prover::backend::metal::quotients::MIN_METAL_QUOTIENT_LOG_SIZE
            {
                let col_slices: Vec<&[BaseField]> = batch
                    .cols_vals_randpows
                    .iter()
                    .map(|data| columns[data.column_index].values.as_slice())
                    .collect();
                let coeff_cs: Vec<SecureField> = coeffs.iter().map(|(_, _, c)| *c).collect();
                let b_sum: SecureField = coeffs.iter().map(|(_, b, _)| *b).sum();
                if let Some(partial_numerators_acc) =
                    crate::prover::backend::metal::quotients::accumulate_numerators_metal(
                        &col_slices,
                        &coeff_cs,
                        -b_sum,
                        subdomain_size,
                    )
                {
                    let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, ..)| a).sum();
                    accumulated_numerators_vec.push(AccumulatedNumerators {
                        sample_point: batch.point,
                        partial_numerators_acc,
                        first_linear_term_acc,
                    });
                    continue;
                }
            }
            let mut partial_numerators_acc =
                unsafe { SecureColumnByCoords::uninitialized(subdomain_size) };

            // Rows are independent; process disjoint row chunks concurrently. Each chunk
            // writes a disjoint row range of every coordinate column.
            let chunk_size = 1 << 12;
            let mut chunk_views = {
                let [c0, c1, c2, c3]: &mut [Vec<BaseField>; 4] =
                    &mut partial_numerators_acc.columns;
                (c0.chunks_mut(chunk_size))
                    .zip(c1.chunks_mut(chunk_size))
                    .zip(c2.chunks_mut(chunk_size))
                    .zip(c3.chunks_mut(chunk_size))
                    .enumerate()
                    .map(|(i, (((d0, d1), d2), d3))| (i * chunk_size, [d0, d1, d2, d3]))
                    .collect_vec()
            };

            // Column-outer accumulation: the row numerator is
            // sum_i (f_i(row) * c_i - b_i) = sum_i f_i(row) * c_i - sum_i b_i, so each
            // batch column is streamed sequentially into per-row accumulators instead of
            // gathering across every column per row. The summands are identical to
            // [`accumulate_row_partial_numerators`].
            let b_sum: SecureField = coeffs.iter().map(|(_, b, _)| *b).sum();
            let process_chunk = |(start, chunk): &mut (usize, [&mut [BaseField]; 4])| {
                use crate::prover::backend::simd::m31::{PackedM31, N_LANES};
                let rows = chunk[0].len();
                if rows.is_multiple_of(N_LANES) {
                    // Packed path: per-coordinate accumulators over the chunk; each
                    // column streams once with one packed load and four broadcast
                    // multiply-accumulates per 16 rows.
                    let n_groups = rows / N_LANES;
                    let neg_coords = (-b_sum).to_m31_array();
                    let mut acc: [Vec<PackedM31>; 4] =
                        neg_coords.map(|coord| vec![PackedM31::broadcast(coord); n_groups]);
                    for (data, (_, _, c)) in zip(&batch.cols_vals_randpows, &coeffs) {
                        let column = columns[data.column_index];
                        let column_chunk = &column[*start..*start + rows];
                        let c_coords = c.to_m31_array().map(PackedM31::broadcast);
                        for (g, group) in column_chunk.chunks_exact(N_LANES).enumerate() {
                            let v = PackedM31::from_array(group.try_into().unwrap());
                            for k in 0..4 {
                                acc[k][g] += v * c_coords[k];
                            }
                        }
                    }
                    for k in 0..4 {
                        for (g, packed) in acc[k].iter().enumerate() {
                            chunk[k][g * N_LANES..(g + 1) * N_LANES]
                                .copy_from_slice(&packed.to_array());
                        }
                    }
                    return;
                }
                let mut acc = vec![-b_sum; rows];
                for (data, (_, _, c)) in zip(&batch.cols_vals_randpows, &coeffs) {
                    let column = columns[data.column_index];
                    let column_chunk = &column[*start..*start + rows];
                    for (a, &v) in acc.iter_mut().zip(column_chunk) {
                        *a += v * *c;
                    }
                }
                for (idx, row_value) in acc.into_iter().enumerate() {
                    let [v0, v1, v2, v3] = row_value.to_m31_array();
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

            let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, ..)| a).sum();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc,
                first_linear_term_acc,
            })
        }
    }

    fn compute_quotients_and_combine(
        accumulations: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let eval_domain = CanonicCoset::new(lifting_log_size).circle_domain();
        let (eval_subdomain, _) = eval_domain.split(log_blowup_factor);
        let subdomain_log_size = eval_subdomain.log_size();

        // Apple-GPU path: the whole per-row combine in one submission (denominator
        // inverses by Fermat exponentiation equal batch_inverse exactly).
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let gpu_quotients = if subdomain_log_size
            >= crate::prover::backend::metal::quotients::MIN_METAL_QUOTIENT_LOG_SIZE
        {
            crate::prover::backend::metal::quotients::combine_quotients_metal(
                &accumulations,
                eval_subdomain,
                1 << subdomain_log_size,
            )
        } else {
            None
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let gpu_quotients: Option<SecureColumnByCoords<CpuBackend>> = None;
        if let Some(quotients) = gpu_quotients {
            return extend_quotients(quotients, eval_subdomain, eval_domain, twiddles);
        }

        let mut quotients: SecureColumnByCoords<CpuBackend> =
            unsafe { SecureColumnByCoords::uninitialized(1 << subdomain_log_size) };
        let sample_points: Vec<CirclePoint<SecureField>> =
            accumulations.iter().map(|x| x.sample_point).collect();
        let n_samples = sample_points.len();
        // Populate `quotients` on the subdomain: rows are independent, so disjoint row
        // chunks are processed concurrently, with the denominator inversions of a whole
        // chunk batched into a single field inversion.
        let chunk_size = 1 << 12;
        let n_rows = quotients.len();
        let mut chunk_views = {
            let [c0, c1, c2, c3]: &mut [Vec<BaseField>; 4] = &mut quotients.columns;
            (c0.chunks_mut(chunk_size))
                .zip(c1.chunks_mut(chunk_size))
                .zip(c2.chunks_mut(chunk_size))
                .zip(c3.chunks_mut(chunk_size))
                .enumerate()
                .map(|(i, (((d0, d1), d2), d3))| (i * chunk_size, [d0, d1, d2, d3]))
                .collect_vec()
        };
        let _ = n_rows;

        let process_chunk = |(start, chunk): &mut (usize, [&mut [BaseField]; 4])| {
            let start = *start;
            let rows = chunk[0].len();
            let mut domain_points = Vec::with_capacity(rows);
            let mut chunk_denominators = Vec::with_capacity(rows * n_samples);
            for idx in 0..rows {
                let domain_point =
                    eval_subdomain.at(bit_reverse_index(start + idx, subdomain_log_size));
                chunk_denominators.extend(denominators(&sample_points, domain_point));
                domain_points.push(domain_point);
            }
            let inverses = CM31::batch_inverse(&chunk_denominators);

            for (idx, &domain_point) in domain_points.iter().enumerate() {
                let row = start + idx;
                let row_inverses = &inverses[idx * n_samples..(idx + 1) * n_samples];
                let mut quotient = SecureField::zero();
                for (acc, &den_inv) in accumulations.iter().zip_eq(row_inverses) {
                    let mut full_numerator = SecureField::zero();
                    let log_ratio = subdomain_log_size - acc.partial_numerators_acc.len().ilog2();
                    let lifted_idx = (row >> (log_ratio + 1) << 1) + (row & 1);

                    full_numerator += acc.partial_numerators_acc.at(lifted_idx)
                        - acc.first_linear_term_acc * domain_point.y;
                    // Note that `den_inv` is an element of CM31 (see the docs and comments in
                    // the function [`crates::core::pcs::quotients::denominator_inverses`]).
                    quotient += full_numerator.mul_cm31(den_inv)
                }
                let [v0, v1, v2, v3] = quotient.to_m31_array();
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
        extend_quotients(quotients, eval_subdomain, eval_domain, twiddles)
    }
}

/// Interpolates the combined quotients on the subdomain and evaluates them on the full
/// lifted domain, shared by the CPU and GPU combine paths.
fn extend_quotients(
    quotients: SecureColumnByCoords<CpuBackend>,
    eval_subdomain: crate::core::poly::circle::CircleDomain,
    eval_domain: crate::core::poly::circle::CircleDomain,
    twiddles: &TwiddleTree<CpuBackend>,
) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
    // Apple-GPU path: all four coordinate transforms (interpolate on the subdomain with
    // its extracted twiddles + evaluate on the lifted domain) in one submission instead
    // of eight synchronized single-column dispatches.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let quotients: SecureColumnByCoords<CpuBackend> = {
        use crate::prover::poly::circle::EvalsOrCoeffs;
        let columns = quotients
            .columns
            .into_iter()
            .map(|column| {
                EvalsOrCoeffs::Evals(
                    CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                        eval_subdomain,
                        column,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let log_blowup = eval_domain.log_size() - eval_subdomain.log_size();
        let subdomain_twiddles = TwiddleTree {
            root_coset: eval_subdomain.half_coset,
            twiddles: TwiddleBuffer::empty(),
            itwiddles: twiddles
                .itwiddles
                .extract_subdomain_twiddles(eval_domain.log_size(), eval_subdomain.log_size()),
        };
        match crate::prover::backend::metal::fft::fused_transform_metal_with_itwiddles(
            columns,
            log_blowup,
            twiddles,
            Some(&subdomain_twiddles),
            false,
        ) {
            Ok(polys) => {
                let mut values = polys.into_iter().map(|poly| poly.evals.values);
                let evals = SecureColumnByCoords {
                    columns: std::array::from_fn(|_| values.next().unwrap()),
                };
                return SecureEvaluation::new(eval_domain, evals);
            }
            Err(columns) => {
                let mut values = columns.into_iter().map(|column| match column {
                    EvalsOrCoeffs::Evals(evals) => evals.values,
                    EvalsOrCoeffs::Coeffs(coeffs) => coeffs.coeffs,
                });
                SecureColumnByCoords {
                    columns: std::array::from_fn(|_| values.next().unwrap()),
                }
            }
        }
    };

    let subdomain_twiddles = TwiddleTree {
        root_coset: eval_subdomain.half_coset,
        twiddles: TwiddleBuffer::empty(),
        itwiddles: twiddles
            .itwiddles
            .extract_subdomain_twiddles(eval_domain.log_size(), eval_subdomain.log_size()),
    };
    let evals = SecureColumnByCoords {
        columns: quotients.columns.map(|eval| {
            let poly = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                eval_subdomain,
                eval,
            )
            .interpolate_with_twiddles(&subdomain_twiddles);
            poly.evaluate_with_twiddles(eval_domain, twiddles).values
        }),
    };
    SecureEvaluation::new(eval_domain, evals)
}
