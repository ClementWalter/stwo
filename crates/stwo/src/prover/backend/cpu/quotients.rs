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
use crate::core::pcs::quotients::{denominators, ColumnSampleBatch};
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::prover::pcs::quotient_ops::{AccumulatedNumerators, NumeratorBatchGroup};
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::twiddles::{TwiddleBuffer, TwiddleTree};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

#[path = "quotients/numerators.rs"]
mod numerators;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn terminal_metal_quotient_result<T>(
    result: Result<Option<T>, crate::prover::backend::metal::quotients::QuotientMetalError>,
) -> Option<T> {
    result.unwrap_or_else(|error| {
        panic!(
            "terminal Metal quotient failure after command submission; CPU fallback is unsafe: \
             {error}"
        )
    })
}

impl QuotientOps for CpuBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    ) {
        numerators::accumulate_group(
            columns,
            sample_batches,
            accumulated_numerators_vec,
            log_blowup_factor,
        );
    }

    fn accumulate_numerator_groups(
        groups: &[NumeratorBatchGroup<'_, Self>],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    ) {
        numerators::accumulate_groups(groups, accumulated_numerators_vec, log_blowup_factor);
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

        // Apple-GPU path: the whole per-row combine in one submission. It batch-inverts
        // denominator norms across 256-row groups with one M31 Fermat inverse per group,
        // then reconstructs the CM31 inverses exactly.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let gpu_quotients = if crate::prover::backend::metal::quotients::should_use_metal_quotients(
            subdomain_log_size,
        ) {
            terminal_metal_quotient_result(
                crate::prover::backend::metal::quotients::combine_quotients_metal(
                    &accumulations,
                    eval_subdomain,
                    1 << subdomain_log_size,
                ),
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

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod metal_error_tests {
    use metal::MTLCommandBufferStatus;

    use super::terminal_metal_quotient_result;
    use crate::prover::backend::metal::quotients::QuotientMetalError;

    #[test]
    fn only_pre_submit_decline_can_fall_back_to_cpu() {
        assert_eq!(terminal_metal_quotient_result::<()>(Ok(None)), None);

        let command_failure = std::panic::catch_unwind(|| {
            terminal_metal_quotient_result::<()>(Err(QuotientMetalError::CommandFailed {
                status: MTLCommandBufferStatus::Error,
            }))
        });
        assert!(command_failure.is_err());

        let numerator_failure = std::panic::catch_unwind(|| {
            terminal_metal_quotient_result::<()>(Err(QuotientMetalError::NumeratorCommandFailed {
                status: MTLCommandBufferStatus::Error,
                batch_count: 14,
                dispatch_count: 42,
            }))
        });
        assert!(numerator_failure.is_err());

        let pole = std::panic::catch_unwind(|| {
            terminal_metal_quotient_result::<()>(Err(QuotientMetalError::Pole {
                sample_mask: 0b10,
                first_row: 17,
                sample_first_rows: [None, Some(17), None, None, None, None],
            }))
        });
        assert!(pole.is_err());
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
