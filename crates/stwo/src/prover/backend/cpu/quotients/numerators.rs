use std::iter::zip;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{quotient_constants, ColumnSampleBatch};
use crate::prover::pcs::quotient_ops::{AccumulatedNumerators, NumeratorBatchGroup};
use crate::prover::poly::circle::CircleEvaluation;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

type LineCoefficients = Vec<(SecureField, SecureField, SecureField)>;

pub(super) fn accumulate_group(
    columns: &[&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>],
    sample_batches: &[ColumnSampleBatch],
    destination: &mut Vec<AccumulatedNumerators<CpuBackend>>,
    log_blowup_factor: u32,
) {
    let subdomain_size = columns[0].len() >> log_blowup_factor;
    let constants = quotient_constants(sample_batches);
    for (batch, coeffs) in zip(sample_batches, constants.line_coeffs) {
        push_accumulation(
            batch,
            &coeffs,
            accumulate_batch_cpu(columns, batch, &coeffs, subdomain_size),
            destination,
        );
    }
}

pub(super) fn accumulate_groups(
    groups: &[NumeratorBatchGroup<'_, CpuBackend>],
    destination: &mut Vec<AccumulatedNumerators<CpuBackend>>,
    log_blowup_factor: u32,
) {
    let constants = groups
        .iter()
        .map(|group| quotient_constants(&group.sample_batches))
        .collect_vec();
    let total_batches = groups.iter().map(|group| group.sample_batches.len()).sum();
    let mut metal_results: Vec<Option<SecureColumnByCoords<CpuBackend>>> =
        std::iter::repeat_with(|| None)
            .take(total_batches)
            .collect();

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use crate::prover::backend::metal::quotients::{
            accumulate_numerators_metal, should_use_metal_numerators, NumeratorBatch,
        };

        let mut eligible_positions = Vec::new();
        let mut metal_batches = Vec::new();
        let mut position = 0usize;
        for (group, constants) in groups.iter().zip(&constants) {
            let subdomain_size = group.columns[0].len() >> log_blowup_factor;
            let selected = should_use_metal_numerators(subdomain_size.ilog2());
            for (batch, coeffs) in group.sample_batches.iter().zip(&constants.line_coeffs) {
                if selected {
                    metal_batches.push(NumeratorBatch {
                        columns: batch
                            .cols_vals_randpows
                            .iter()
                            .map(|data| group.columns[data.column_index].values.as_slice())
                            .collect(),
                        coeffs: coeffs.iter().map(|(_, _, c)| *c).collect(),
                        neg_b_sum: -coeffs.iter().map(|(_, b, _)| *b).sum::<SecureField>(),
                        n_rows: subdomain_size,
                    });
                    eligible_positions.push(position);
                }
                position += 1;
            }
        }
        if !metal_batches.is_empty() {
            if let Some(outputs) =
                super::terminal_metal_quotient_result(accumulate_numerators_metal(&metal_batches))
            {
                assert_eq!(outputs.len(), eligible_positions.len());
                for (position, output) in eligible_positions.into_iter().zip(outputs) {
                    metal_results[position] = Some(output);
                }
            }
        }
    }

    // Reconstruct the exact former append order: ascending log group, stable point
    // batch. Ineligible work and a pre-submit whole-command decline replay on CPU.
    let mut position = 0usize;
    for (group, constants) in groups.iter().zip(constants) {
        let subdomain_size = group.columns[0].len() >> log_blowup_factor;
        for (batch, coeffs) in group.sample_batches.iter().zip(constants.line_coeffs) {
            let partial_numerators_acc = metal_results[position].take().unwrap_or_else(|| {
                accumulate_batch_cpu(&group.columns, batch, &coeffs, subdomain_size)
            });
            push_accumulation(batch, &coeffs, partial_numerators_acc, destination);
            position += 1;
        }
    }
}

fn push_accumulation(
    batch: &ColumnSampleBatch,
    coeffs: &LineCoefficients,
    partial_numerators_acc: SecureColumnByCoords<CpuBackend>,
    destination: &mut Vec<AccumulatedNumerators<CpuBackend>>,
) {
    destination.push(AccumulatedNumerators {
        sample_point: batch.point,
        partial_numerators_acc,
        first_linear_term_acc: coeffs.iter().map(|(a, ..)| a).sum(),
    });
}

fn accumulate_batch_cpu(
    columns: &[&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>],
    batch: &ColumnSampleBatch,
    coeffs: &LineCoefficients,
    subdomain_size: usize,
) -> SecureColumnByCoords<CpuBackend> {
    // SAFETY: the chunk views below partition every coordinate into disjoint slices
    // covering all `subdomain_size` rows. Both processing branches initialize every
    // element of all four slices before `output` is read or returned.
    let mut output = unsafe { SecureColumnByCoords::uninitialized(subdomain_size) };
    let chunk_size = 1 << 12;
    let mut chunks = {
        let [c0, c1, c2, c3]: &mut [Vec<BaseField>; 4] = &mut output.columns;
        c0.chunks_mut(chunk_size)
            .zip(c1.chunks_mut(chunk_size))
            .zip(c2.chunks_mut(chunk_size))
            .zip(c3.chunks_mut(chunk_size))
            .enumerate()
            .map(|(index, (((d0, d1), d2), d3))| (index * chunk_size, [d0, d1, d2, d3]))
            .collect_vec()
    };
    let b_sum: SecureField = coeffs.iter().map(|(_, b, _)| *b).sum();
    let process_chunk = |(start, chunk): &mut (usize, [&mut [BaseField]; 4])| {
        use crate::prover::backend::simd::m31::{PackedM31, N_LANES};
        let rows = chunk[0].len();
        if rows.is_multiple_of(N_LANES) {
            let n_groups = rows / N_LANES;
            let neg_coords = (-b_sum).to_m31_array();
            let mut accumulators: [Vec<PackedM31>; 4] =
                neg_coords.map(|coord| vec![PackedM31::broadcast(coord); n_groups]);
            for (data, (_, _, coefficient)) in zip(&batch.cols_vals_randpows, coeffs) {
                let values = &columns[data.column_index][*start..*start + rows];
                let packed_coefficients = coefficient.to_m31_array().map(PackedM31::broadcast);
                for (group, values) in values.chunks_exact(N_LANES).enumerate() {
                    let value = PackedM31::from_array(values.try_into().unwrap());
                    for coordinate in 0..4 {
                        accumulators[coordinate][group] += value * packed_coefficients[coordinate];
                    }
                }
            }
            for coordinate in 0..4 {
                for (group, packed) in accumulators[coordinate].iter().enumerate() {
                    chunk[coordinate][group * N_LANES..(group + 1) * N_LANES]
                        .copy_from_slice(&packed.to_array());
                }
            }
            return;
        }
        let mut accumulators = vec![-b_sum; rows];
        for (data, (_, _, coefficient)) in zip(&batch.cols_vals_randpows, coeffs) {
            let values = &columns[data.column_index][*start..*start + rows];
            for (accumulator, &value) in accumulators.iter_mut().zip(values) {
                *accumulator += value * *coefficient;
            }
        }
        for (row, value) in accumulators.into_iter().enumerate() {
            let [v0, v1, v2, v3] = value.to_m31_array();
            chunk[0][row] = v0;
            chunk[1][row] = v1;
            chunk[2][row] = v2;
            chunk[3][row] = v3;
        }
    };
    #[cfg(feature = "parallel")]
    chunks.par_iter_mut().for_each(process_chunk);
    #[cfg(not(feature = "parallel"))]
    chunks.iter_mut().for_each(process_chunk);
    output
}
