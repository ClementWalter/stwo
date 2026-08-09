use metal::{Buffer, MTLResourceOptions, MTLSize};

use super::{bind_input, bind_output, context, CHUNK_COLS};
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

/// GPU partial-numerator accumulation over the first `n_rows` entries of each column:
/// `acc[row] = -b_sum + sum_i columns[i][row] * coeffs[i]`. Returns `None` when no
/// usable device exists or command execution fails.
pub(crate) fn accumulate_numerators_metal(
    columns: &[&[BaseField]],
    coeffs: &[SecureField],
    neg_b_sum: SecureField,
    n_rows: usize,
) -> Option<SecureColumnByCoords<CpuBackend>> {
    assert_eq!(columns.len(), coeffs.len());
    let ctx = context()?;
    let mut ctx = ctx.lock().unwrap();

    let mut out: [Vec<BaseField>; 4] = std::array::from_fn(|_| vec![BaseField::default(); n_rows]);
    let state_bytes = (n_rows * 16) as u64;
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

    let input_buffers: Vec<Buffer> = columns
        .iter()
        .map(|column| bind_input(&ctx.device, &column[..n_rows.min(column.len())]))
        .collect();
    let output_buffers: Vec<Buffer> = out
        .iter_mut()
        .map(|column| bind_output(&ctx.device, column))
        .collect();
    let command_buffer = ctx.queue.new_command_buffer();
    let n_chunks = columns.len().div_ceil(CHUNK_COLS);
    for (chunk_index, (buffer_chunk, coefficient_chunk)) in input_buffers
        .chunks(CHUNK_COLS)
        .zip(coeffs.chunks(CHUNK_COLS))
        .enumerate()
    {
        let mut packed_coefficients = [0u32; 64];
        for (column, coefficient) in coefficient_chunk.iter().enumerate() {
            let coordinates = coefficient.to_m31_array();
            for coordinate in 0..4 {
                packed_coefficients[column * 4 + coordinate] = coordinates[coordinate].0;
            }
        }
        let params = AccumulateParams {
            n_rows: n_rows as u32,
            n_cols: buffer_chunk.len() as u32,
            mode: u32::from(chunk_index == 0) | (u32::from(chunk_index == n_chunks - 1) << 1),
            init_acc: neg_b_sum.to_m31_array().map(|value| value.0),
            coeffs: packed_coefficients,
        };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        for slot in 0..CHUNK_COLS {
            // Unused slots remain bound but alias an input the kernel never reads.
            let buffer = buffer_chunk.get(slot).unwrap_or(&buffer_chunk[0]);
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_buffer(16, ctx.state_buffer.as_ref().map(|buffer| buffer as _), 0);
        for (coordinate, buffer) in output_buffers.iter().enumerate() {
            encoder.set_buffer(17 + coordinate as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            21,
            std::mem::size_of::<AccumulateParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }
    command_buffer.commit();
    super::super::context::wait_for_completion(command_buffer).ok()?;

    for (column, buffer) in out.iter_mut().zip(&output_buffers) {
        let zero_copy = std::ptr::eq(buffer.contents() as *const BaseField, column.as_ptr());
        if !zero_copy {
            // SAFETY: successful checked completion initialized exactly `n_rows`
            // u32-aligned BaseField words in this live StorageModeShared buffer. The
            // non-zero-copy branch guarantees source and destination do not overlap;
            // both allocations outlive the copy and the GPU no longer mutates either.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const BaseField,
                    column.as_mut_ptr(),
                    n_rows,
                );
            }
        }
    }
    let [c0, c1, c2, c3] = out;
    Some(SecureColumnByCoords {
        columns: [c0, c1, c2, c3],
    })
}
