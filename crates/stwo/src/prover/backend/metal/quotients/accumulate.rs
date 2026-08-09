use std::collections::BTreeMap;

use metal::{Buffer, MTLResourceOptions, MTLSize};

use super::{bind_input, bind_output, context, QuotientMetalError, CHUNK_COLS};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;

#[repr(C)]
struct AccumulateParams {
    n_rows: u32,
    n_cols: u32,
    mode: u32,
    init_acc: [u32; 4],
    coeffs: [u32; 64],
}

// ABI pin: this layout must stay byte-for-byte identical to `AccumulateParams` in
// `accumulate.metal`, because `set_bytes` copies this value directly into buffer 21.
const _: () = {
    assert!(std::mem::size_of::<AccumulateParams>() == 284);
    assert!(std::mem::align_of::<AccumulateParams>() == 4);
    assert!(std::mem::offset_of!(AccumulateParams, n_rows) == 0);
    assert!(std::mem::offset_of!(AccumulateParams, n_cols) == 4);
    assert!(std::mem::offset_of!(AccumulateParams, mode) == 8);
    assert!(std::mem::offset_of!(AccumulateParams, init_acc) == 12);
    assert!(std::mem::offset_of!(AccumulateParams, coeffs) == 28);
};

/// One stable point batch after columns have been grouped by ascending evaluation log.
pub(crate) struct NumeratorBatch<'a> {
    pub columns: Vec<&'a [BaseField]>,
    pub coeffs: Vec<SecureField>,
    pub neg_b_sum: SecureField,
    pub n_rows: usize,
}

struct AccumulationPlan {
    max_rows: usize,
    dispatch_count: usize,
}

fn input_key(column: &[BaseField]) -> (usize, usize) {
    (column.as_ptr() as usize, column.len())
}

/// Performs every validation before a command buffer exists. `None` is therefore a
/// mutation-free decline that is safe to replay on the packed CPU path.
fn preflight(batches: &[NumeratorBatch<'_>]) -> Option<AccumulationPlan> {
    if batches.is_empty() {
        return None;
    }
    let mut max_rows = 0;
    let mut dispatch_count = 0usize;
    for batch in batches {
        let valid = batch.n_rows != 0
            && batch.n_rows.is_power_of_two()
            && batch.n_rows <= u32::MAX as usize
            && !batch.columns.is_empty()
            && batch.columns.len() == batch.coeffs.len()
            && batch
                .columns
                .iter()
                .all(|column| column.len() >= batch.n_rows)
            && batch
                .n_rows
                .checked_mul(4 * std::mem::size_of::<BaseField>())
                .is_some();
        if !valid {
            return None;
        }
        max_rows = max_rows.max(batch.n_rows);
        dispatch_count = dispatch_count.checked_add(batch.columns.len().div_ceil(CHUNK_COLS))?;
    }
    Some(AccumulationPlan {
        max_rows,
        dispatch_count,
    })
}

/// GPU partial-numerator accumulation for all eligible cross-log batches:
/// `acc[row] = -b_sum + sum_i columns[i][row] * coeffs[i]`.
///
/// Batches, and the 16-column chunks within each batch, are encoded in caller order in
/// one command buffer. Metal executes those compute encoders in order, so one shared
/// running-state buffer is safe: each batch's first chunk initializes every active row
/// before a later chunk can read it. Inputs are globally deduplicated by exact pointer
/// and bound length. `Ok(None)` is possible only before submission; every failed checked
/// wait after submission is terminal.
pub(crate) fn accumulate_numerators_metal(
    batches: &[NumeratorBatch<'_>],
) -> Result<Option<Vec<SecureColumnByCoords<CpuBackend>>>, QuotientMetalError> {
    let Some(plan) = preflight(batches) else {
        return Ok(None);
    };
    let Some(ctx) = context() else {
        return Ok(None);
    };
    let Ok(mut ctx) = ctx.lock() else {
        return Ok(None);
    };

    let state_bytes = (plan.max_rows * 4 * std::mem::size_of::<BaseField>()) as u64;
    if ctx
        .state_buffer
        .as_ref()
        .is_none_or(|buffer| buffer.length() < state_bytes)
    {
        ctx.state_buffer = Some(
            ctx.device
                .new_buffer(state_bytes, MTLResourceOptions::StorageModePrivate),
        );
    }

    let mut input_buffers = Vec::<Buffer>::new();
    let mut input_by_address = BTreeMap::<(usize, usize), usize>::new();
    let mut batch_input_indices = Vec::with_capacity(batches.len());
    for batch in batches {
        let mut indices = Vec::with_capacity(batch.columns.len());
        for column in &batch.columns {
            let slice = &column[..batch.n_rows];
            let key = input_key(slice);
            let buffer_index = match input_by_address.get(&key) {
                Some(&index) => index,
                None => {
                    let index = input_buffers.len();
                    input_buffers.push(bind_input(&ctx.device, slice));
                    input_by_address.insert(key, index);
                    index
                }
            };
            indices.push(buffer_index);
        }
        batch_input_indices.push(indices);
    }

    let mut outputs: Vec<[Vec<BaseField>; 4]> = batches
        .iter()
        .map(|batch| std::array::from_fn(|_| vec![BaseField::default(); batch.n_rows]))
        .collect();
    let output_buffers: Vec<[Buffer; 4]> = outputs
        .iter_mut()
        .map(|output| {
            let [c0, c1, c2, c3] = output;
            [
                bind_output(&ctx.device, c0),
                bind_output(&ctx.device, c1),
                bind_output(&ctx.device, c2),
                bind_output(&ctx.device, c3),
            ]
        })
        .collect();

    let command_buffer = ctx.queue.new_command_buffer();
    for ((batch, input_indices), batch_outputs) in batches
        .iter()
        .zip(&batch_input_indices)
        .zip(&output_buffers)
    {
        let n_chunks = input_indices.len().div_ceil(CHUNK_COLS);
        for (chunk_index, (index_chunk, coefficient_chunk)) in input_indices
            .chunks(CHUNK_COLS)
            .zip(batch.coeffs.chunks(CHUNK_COLS))
            .enumerate()
        {
            let mut packed_coefficients = [0u32; 64];
            for (column, coefficient) in coefficient_chunk.iter().enumerate() {
                for (coordinate, value) in coefficient.to_m31_array().iter().enumerate() {
                    packed_coefficients[column * 4 + coordinate] = value.0;
                }
            }
            let params = AccumulateParams {
                n_rows: batch.n_rows as u32,
                n_cols: index_chunk.len() as u32,
                mode: u32::from(chunk_index == 0) | (u32::from(chunk_index == n_chunks - 1) << 1),
                init_acc: batch.neg_b_sum.to_m31_array().map(|value| value.0),
                coeffs: packed_coefficients,
            };
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&ctx.pipeline);
            for slot in 0..CHUNK_COLS {
                let index = *index_chunk.get(slot).unwrap_or(&index_chunk[0]);
                encoder.set_buffer(slot as u64, Some(&input_buffers[index]), 0);
            }
            encoder.set_buffer(16, ctx.state_buffer.as_ref().map(|buffer| buffer as _), 0);
            for (coordinate, buffer) in batch_outputs.iter().enumerate() {
                encoder.set_buffer(17 + coordinate as u64, Some(buffer), 0);
            }
            encoder.set_bytes(
                21,
                std::mem::size_of::<AccumulateParams>() as u64,
                &params as *const _ as *const std::ffi::c_void,
            );
            encoder.dispatch_threads(
                MTLSize::new(batch.n_rows as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
        }
    }
    command_buffer.commit();
    super::super::context::wait_for_completion(command_buffer).map_err(|status| {
        QuotientMetalError::NumeratorCommandFailed {
            status,
            batch_count: batches.len(),
            dispatch_count: plan.dispatch_count,
        }
    })?;

    // Keep every no-copy input and every output allocation alive through the checked
    // wait. From this point the GPU can no longer read or mutate them.
    let _retained_inputs = &input_buffers;
    for (output, buffers) in outputs.iter_mut().zip(&output_buffers) {
        for (column, buffer) in output.iter_mut().zip(buffers) {
            let zero_copy = std::ptr::eq(buffer.contents() as *const BaseField, column.as_ptr());
            if !zero_copy {
                // SAFETY: checked completion initialized exactly `column.len()` aligned
                // BaseField words. Source and destination do not overlap in this branch,
                // both allocations are live, and the GPU has stopped mutating them.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.contents() as *const BaseField,
                        column.as_mut_ptr(),
                        column.len(),
                    );
                }
            }
        }
    }
    Ok(Some(
        outputs
            .into_iter()
            .map(|columns| SecureColumnByCoords { columns })
            .collect(),
    ))
}

#[cfg(all(test, feature = "parallel"))]
pub(super) fn accumulate_numerators_metal_serial(
    batch: &NumeratorBatch<'_>,
) -> Result<Option<SecureColumnByCoords<CpuBackend>>, QuotientMetalError> {
    Ok(accumulate_numerators_metal(std::slice::from_ref(batch))?
        .map(|mut outputs| outputs.pop().unwrap()))
}

#[cfg(test)]
pub(super) fn preflight_for_test(batches: &[NumeratorBatch<'_>]) -> Option<(usize, usize)> {
    preflight(batches).map(|plan| (plan.max_rows, plan.dispatch_count))
}

#[cfg(test)]
pub(super) fn input_key_for_test(column: &[BaseField]) -> (usize, usize) {
    input_key(column)
}
