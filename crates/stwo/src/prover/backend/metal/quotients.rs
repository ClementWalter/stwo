//! Apple-GPU quotient numerator accumulation.
//!
//! Per sample batch, the partial numerator at a row is
//! `-sum_i(b_i) + sum_i(f_i(row) * c_i)` with `c_i` a QM31 constant per column, so a
//! GPU thread accumulates four M31 coordinate sums per row. Columns are streamed in
//! chunks of 16 buffers per dispatch (Metal binding limit) with a running QM31 state
//! per row, like the blake2s leaf builder. Output equals the CPU column-outer
//! accumulation exactly (field-associative regrouping only).

use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::core::fields::m31::BaseField;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;

/// Minimum rows for GPU dispatch.
pub(crate) const MIN_METAL_QUOTIENT_LOG_SIZE: u32 = 16;

const CHUNK_COLS: usize = 16;

const KERNEL_SOURCE: &str = include_str!("quotients/accumulate.metal");
const COMBINE_KERNEL: &str = include_str!("quotients/combine_batched.metal");
#[cfg(test)]
const DIRECT_COMBINE_KERNEL: &str = include_str!("quotients/combine_direct.metal");

#[path = "quotients/error.rs"]
mod error;
pub(crate) use error::QuotientMetalError;
#[path = "quotients/accumulate.rs"]
mod accumulate;
#[path = "quotients/domain.rs"]
mod domain;
pub(crate) use accumulate::accumulate_numerators_metal;

struct QuotientContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    combine_pipeline: Option<ComputePipelineState>,
    #[cfg(test)]
    direct_combine_pipeline: Option<ComputePipelineState>,
    /// Reused running-state buffer (n_rows x 4 u32), grown on demand.
    state_buffer: Option<Buffer>,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for QuotientContext {}

fn context() -> Option<&'static Mutex<QuotientContext>> {
    static CONTEXT: OnceLock<Option<Mutex<QuotientContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let shared = super::context::gpu()?;
            let device = shared.device.clone();
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let function = library.get_function("accumulate_numerators", None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            let queue = shared.queue.clone();
            Some(Mutex::new(QuotientContext {
                device,
                queue,
                pipeline,
                combine_pipeline: None,
                #[cfg(test)]
                direct_combine_pipeline: None,
                state_buffer: None,
            }))
        })
        .as_ref()
}

/// Forces device/pipeline initialization; see [`super::warmup`].
pub(crate) fn warmup() {
    let _ = context();
}

/// Returns whether both static quotient pipelines compiled successfully.
pub(crate) fn is_ready() -> bool {
    let Some(context) = context() else {
        return false;
    };
    let Ok(mut context) = context.lock() else {
        return false;
    };
    ensure_combine_pipeline(&mut context).is_some()
}

fn bind_input(device: &Device, data: &[BaseField]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let page = 16384;
    if (data.as_ptr() as usize).is_multiple_of(page) && bytes.is_multiple_of(page) {
        // The caller keeps the data alive until the awaited command buffer completes.
        device.new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const std::ffi::c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    } else {
        device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

fn bind_output(device: &Device, data: &mut [BaseField]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let page = 16384;
    if (data.as_mut_ptr() as usize).is_multiple_of(page) && bytes.is_multiple_of(page) {
        // The exclusive borrow is not used again until the awaited command completes.
        device.new_buffer_with_bytes_no_copy(
            data.as_mut_ptr() as *const std::ffi::c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    } else {
        device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    }
}

fn ensure_combine_pipeline(context: &mut QuotientContext) -> Option<()> {
    if context.combine_pipeline.is_some() {
        return Some(());
    }
    let library = context
        .device
        .new_library_with_source(COMBINE_KERNEL, &CompileOptions::new())
        .ok()?;
    let function = library
        .get_function("combine_quotients_batched", None)
        .ok()?;
    let pipeline = context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .ok()?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return None;
    }
    context.combine_pipeline = Some(pipeline);
    Some(())
}

#[cfg(test)]
fn ensure_direct_combine_pipeline(context: &mut QuotientContext) -> Option<()> {
    if context.direct_combine_pipeline.is_some() {
        return Some(());
    }
    let library = context
        .device
        .new_library_with_source(DIRECT_COMBINE_KERNEL, &CompileOptions::new())
        .ok()?;
    let function = library
        .get_function("combine_quotients_direct", None)
        .ok()?;
    let pipeline = context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .ok()?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return None;
    }
    context.direct_combine_pipeline = Some(pipeline);
    Some(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SampleParams {
    prx: [u32; 2],
    pix: [u32; 2],
    pry: [u32; 2],
    piy: [u32; 2],
    flt0: [u32; 2],
    flt1: [u32; 2],
    log_ratio: u32,
    _pad: u32,
}

#[repr(C)]
struct CombineParams {
    n_rows: u32,
    n_samples: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CombineMode {
    Batched,
    #[cfg(test)]
    Direct,
}

/// GPU quotient combine over the subdomain. Denominator norms are batch-inverted with
/// one M31 Fermat inverse per 256-row threadgroup, then each CM31 inverse and numerator
/// is recovered in canonical sample order. `Ok(None)` is a pre-submit decline. Every
/// post-submit command failure or denominator pole is a terminal [`QuotientMetalError`].
pub(crate) fn combine_quotients_metal(
    accumulations: &[crate::prover::pcs::quotient_ops::AccumulatedNumerators<CpuBackend>],
    subdomain: crate::core::poly::circle::CircleDomain,
    n_rows: usize,
) -> Result<Option<SecureColumnByCoords<CpuBackend>>, QuotientMetalError> {
    let xy = domain::get(subdomain);
    combine_quotients_metal_with_xy(accumulations, subdomain.log_size(), n_rows, &xy.xs, &xy.ys)
}

fn combine_quotients_metal_with_xy(
    accumulations: &[crate::prover::pcs::quotient_ops::AccumulatedNumerators<CpuBackend>],
    subdomain_log_size: u32,
    n_rows: usize,
    xs: &[BaseField],
    ys: &[BaseField],
) -> Result<Option<SecureColumnByCoords<CpuBackend>>, QuotientMetalError> {
    combine_quotients_metal_impl(
        accumulations,
        subdomain_log_size,
        n_rows,
        xs,
        ys,
        CombineMode::Batched,
    )
}

#[cfg(test)]
fn combine_quotients_direct_metal(
    accumulations: &[crate::prover::pcs::quotient_ops::AccumulatedNumerators<CpuBackend>],
    subdomain: crate::core::poly::circle::CircleDomain,
) -> Result<Option<SecureColumnByCoords<CpuBackend>>, QuotientMetalError> {
    let xy = domain::get(subdomain);
    combine_quotients_metal_impl(
        accumulations,
        subdomain.log_size(),
        subdomain.size(),
        &xy.xs,
        &xy.ys,
        CombineMode::Direct,
    )
}

fn combine_quotients_metal_impl(
    accumulations: &[crate::prover::pcs::quotient_ops::AccumulatedNumerators<CpuBackend>],
    subdomain_log_size: u32,
    n_rows: usize,
    xs: &[BaseField],
    ys: &[BaseField],
    mode: CombineMode,
) -> Result<Option<SecureColumnByCoords<CpuBackend>>, QuotientMetalError> {
    use metal::MTLResourceUsage;
    let Some(domain_rows) = 1usize.checked_shl(subdomain_log_size) else {
        return Ok(None);
    };
    let shape_is_valid = !accumulations.is_empty()
        && accumulations.len() <= 6
        && n_rows != 0
        && n_rows <= u32::MAX as usize
        && n_rows <= domain_rows
        && xs.len() >= n_rows
        && ys.len() >= n_rows
        && accumulations.iter().all(|accumulation| {
            let len = accumulation.partial_numerators_acc.len();
            len != 0
                && len.is_power_of_two()
                && len <= domain_rows
                && accumulation
                    .partial_numerators_acc
                    .columns
                    .iter()
                    .all(|column| column.len() == len)
        });
    if !shape_is_valid {
        return Ok(None);
    }
    let Some(ctx) = context() else {
        return Ok(None);
    };
    let Ok(mut ctx) = ctx.lock() else {
        return Ok(None);
    };
    let pipeline_is_ready = match mode {
        CombineMode::Batched => ensure_combine_pipeline(&mut ctx).is_some(),
        #[cfg(test)]
        CombineMode::Direct => ensure_direct_combine_pipeline(&mut ctx).is_some(),
    };
    if !pipeline_is_ready {
        return Ok(None);
    }

    let samples: Vec<SampleParams> = accumulations
        .iter()
        .map(|acc| {
            let to_pair = |v: crate::core::fields::cm31::CM31| [v.0 .0, v.1 .0];
            let flt = acc.first_linear_term_acc.to_m31_array();
            SampleParams {
                prx: to_pair(acc.sample_point.x.0),
                pix: to_pair(acc.sample_point.x.1),
                pry: to_pair(acc.sample_point.y.0),
                piy: to_pair(acc.sample_point.y.1),
                flt0: [flt[0].0, flt[1].0],
                flt1: [flt[2].0, flt[3].0],
                log_ratio: subdomain_log_size - acc.partial_numerators_acc.len().ilog2(),
                _pad: 0,
            }
        })
        .collect();

    let mut acc_buffers = Vec::new();
    let mut addresses: Vec<u64> = Vec::new();
    for acc in accumulations {
        for column in &acc.partial_numerators_acc.columns {
            let buffer = bind_input(&ctx.device, column);
            addresses.push(buffer.gpu_address());
            acc_buffers.push(buffer);
        }
    }
    let addr_buffer = ctx.device.new_buffer_with_data(
        addresses.as_ptr() as *const std::ffi::c_void,
        (addresses.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let xs_buffer = bind_input(&ctx.device, &xs[..n_rows]);
    let ys_buffer = bind_input(&ctx.device, &ys[..n_rows]);

    let mut out: [Vec<BaseField>; 4] = std::array::from_fn(|_| vec![BaseField::default(); n_rows]);
    let out_buffers: Vec<Buffer> = out
        .iter_mut()
        .map(|c| bind_output(&ctx.device, c))
        .collect();
    let initial_pole_report = [
        0,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
    ];
    let pole_report_buffer = ctx.device.new_buffer_with_data(
        initial_pole_report.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(&initial_pole_report) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let params = CombineParams {
        n_rows: n_rows as u32,
        n_samples: samples.len() as u32,
    };
    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let combine_pipeline = match mode {
        CombineMode::Batched => ctx.combine_pipeline.as_ref().unwrap(),
        #[cfg(test)]
        CombineMode::Direct => ctx.direct_combine_pipeline.as_ref().unwrap(),
    };
    encoder.set_compute_pipeline_state(combine_pipeline);
    encoder.set_buffer(0, Some(&addr_buffer), 0);
    encoder.set_buffer(1, Some(&xs_buffer), 0);
    encoder.set_buffer(2, Some(&ys_buffer), 0);
    for (k, buffer) in out_buffers.iter().enumerate() {
        encoder.set_buffer(3 + k as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        7,
        std::mem::size_of_val(samples.as_slice()) as u64,
        samples.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<CombineParams>() as u64,
        &params as *const _ as *const std::ffi::c_void,
    );
    if mode == CombineMode::Batched {
        encoder.set_buffer(9, Some(&pole_report_buffer), 0);
    }
    for buffer in &acc_buffers {
        encoder.use_resource(buffer, MTLResourceUsage::Read);
    }
    if mode == CombineMode::Batched {
        let n_threadgroups = n_rows.div_ceil(256);
        encoder.dispatch_thread_groups(
            MTLSize::new(n_threadgroups as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    } else {
        encoder.dispatch_threads(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }
    encoder.end_encoding();
    command_buffer.commit();
    super::context::wait_for_completion(command_buffer)
        .map_err(|status| QuotientMetalError::CommandFailed { status })?;

    // SAFETY: `pole_report_buffer` is a live, non-empty StorageModeShared allocation
    // of exactly seven u32 words, and Metal buffer contents are u32-aligned. Checked
    // command completion makes all GPU atomic writes visible before this synchronous
    // read. The buffer handle outlives the borrowed slice, and neither host nor GPU
    // mutates the allocation while the slice is consumed.
    let pole_report =
        unsafe { std::slice::from_raw_parts(pole_report_buffer.contents() as *const u32, 7) };
    let sample_mask = if mode == CombineMode::Batched {
        pole_report[0]
    } else {
        0
    };
    if sample_mask != 0 {
        let sample_first_rows = std::array::from_fn(|sample| {
            (pole_report[1 + sample] != u32::MAX).then_some(pole_report[1 + sample] as usize)
        });
        let first_row = sample_first_rows
            .iter()
            .flatten()
            .copied()
            .min()
            .unwrap_or(usize::MAX);
        return Err(QuotientMetalError::Pole {
            sample_mask,
            first_row,
            sample_first_rows,
        });
    }

    for (vec, buffer) in out.iter_mut().zip(&out_buffers) {
        let zero_copy = std::ptr::eq(buffer.contents() as *const BaseField, vec.as_ptr());
        if !zero_copy {
            // SAFETY: successful checked completion initialized exactly `n_rows`
            // u32-aligned BaseField words in this live StorageModeShared buffer. The
            // non-zero-copy branch guarantees source and destination do not overlap;
            // both allocations outlive the copy and the GPU no longer mutates either.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const BaseField,
                    vec.as_mut_ptr(),
                    n_rows,
                );
            }
        }
    }
    let [c0, c1, c2, c3] = out;
    Ok(Some(SecureColumnByCoords {
        columns: [c0, c1, c2, c3],
    }))
}

#[cfg(test)]
#[path = "quotients_tests.rs"]
mod tests;
