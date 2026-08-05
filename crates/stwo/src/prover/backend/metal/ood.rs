//! Apple-GPU out-of-domain sampling: the dot product of each polynomial's
//! coefficients with the shared FFT basis column.
//!
//! One threadgroup per polynomial: threads accumulate strided partial sums of
//! `coeff * basis` per QM31 coordinate (coefficients are M31, so each coordinate is an
//! independent M31 dot), then tree-reduce in threadgroup memory. Field addition is
//! associative and commutative, so the regrouped sum equals the CPU fold exactly.

use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::CpuBackend;
use crate::prover::poly::circle::CircleCoefficients;

/// Minimum coefficient count for GPU dispatch.
pub(crate) const MIN_METAL_OOD_LOG_SIZE: u32 = 16;

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint P = 0x7FFFFFFFu;

inline uint m31_add(uint a, uint b) {
    uint s = a + b;
    return (s >= P) ? s - P : s;
}

inline uint m31_mul(uint a, uint b) {
    ulong p = (ulong)a * (ulong)b;
    uint s = (uint)(p & P) + (uint)(p >> 31);
    s = (s & P) + (s >> 31);
    return (s >= P) ? s - P : s;
}

struct ColPtr { device const uint* data; };

kernel void ood_dots(
    device const ColPtr* cols [[buffer(0)]],
    device const uint* basis [[buffer(1)]],   // QM31 AoS: 4 u32 per row
    device uint* out [[buffer(2)]],           // 4 u32 per column
    constant uint& n_rows [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint col [[threadgroup_position_in_grid]])
{
    device const uint* coeffs = cols[col].data;
    uint acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (uint i = tid; i < n_rows; i += 256u) {
        uint c = coeffs[i];
        acc0 = m31_add(acc0, m31_mul(c, basis[i * 4u + 0u]));
        acc1 = m31_add(acc1, m31_mul(c, basis[i * 4u + 1u]));
        acc2 = m31_add(acc2, m31_mul(c, basis[i * 4u + 2u]));
        acc3 = m31_add(acc3, m31_mul(c, basis[i * 4u + 3u]));
    }
    threadgroup uint partial[256 * 4];
    partial[tid * 4u + 0u] = acc0;
    partial[tid * 4u + 1u] = acc1;
    partial[tid * 4u + 2u] = acc2;
    partial[tid * 4u + 3u] = acc3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            for (uint k = 0; k < 4u; k++) {
                partial[tid * 4u + k] =
                    m31_add(partial[tid * 4u + k], partial[(tid + stride) * 4u + k]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        for (uint k = 0; k < 4u; k++) { out[col * 4u + k] = partial[k]; }
    }
}
"#;

struct OodContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for OodContext {}

fn context() -> Option<&'static Mutex<OodContext>> {
    static CONTEXT: OnceLock<Option<Mutex<OodContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let shared = super::context::gpu()?;
            let device = shared.device.clone();
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let function = library.get_function("ood_dots", None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            let queue = shared.queue.clone();
            Some(Mutex::new(OodContext {
                device,
                queue,
                pipeline,
            }))
        })
        .as_ref()
}

/// Forces device/pipeline initialization; see [`super::warmup`].
pub(crate) fn warmup() {
    let _ = context();
}

fn bind_input(device: &Device, ptr: *const u8, bytes: usize) -> Buffer {
    let page = 16384;
    if (ptr as usize).is_multiple_of(page) && bytes.is_multiple_of(page) {
        // The caller keeps the data alive until the awaited command buffer completes.
        device.new_buffer_with_bytes_no_copy(
            ptr as *const std::ffi::c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    } else {
        device.new_buffer_with_data(
            ptr as *const std::ffi::c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

/// GPU shared-basis evaluation of many same-size polynomials at one point; equals
/// mapping the CPU dot exactly (summation regrouping only). Returns `None` without a
/// usable device or successful command.
pub(crate) fn eval_many_metal(
    polys: &[&CircleCoefficients<CpuBackend>],
    basis: &[SecureField],
) -> Option<Vec<SecureField>> {
    let n_rows = basis.len();
    if n_rows < (1 << MIN_METAL_OOD_LOG_SIZE) || polys.is_empty() {
        return None;
    }
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    let mut col_buffers = Vec::with_capacity(polys.len());
    let mut addresses: Vec<u64> = Vec::with_capacity(polys.len());
    for poly in polys {
        assert_eq!(poly.coeffs.len(), n_rows);
        let buffer = bind_input(&ctx.device, poly.coeffs.as_ptr() as *const u8, n_rows * 4);
        addresses.push(buffer.gpu_address());
        col_buffers.push(buffer);
    }
    let addr_buffer = ctx.device.new_buffer_with_data(
        addresses.as_ptr() as *const std::ffi::c_void,
        (addresses.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let basis_buffer = bind_input(&ctx.device, basis.as_ptr() as *const u8, n_rows * 16);
    let out_buffer = ctx.device.new_buffer(
        (polys.len() * 16) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&ctx.pipeline);
    encoder.set_buffer(0, Some(&addr_buffer), 0);
    encoder.set_buffer(1, Some(&basis_buffer), 0);
    encoder.set_buffer(2, Some(&out_buffer), 0);
    let rows = n_rows as u32;
    encoder.set_bytes(3, 4, &rows as *const _ as *const std::ffi::c_void);
    for buffer in &col_buffers {
        encoder.use_resource(buffer, MTLResourceUsage::Read);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(polys.len() as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    super::context::wait_for_completion(command_buffer).ok()?;

    // Safety: the kernel wrote 4 coordinates per column.
    let words =
        unsafe { std::slice::from_raw_parts(out_buffer.contents() as *const u32, polys.len() * 4) };
    Some(
        words
            .chunks_exact(4)
            .map(|c| {
                SecureField::from_m31_array(std::array::from_fn(|k| {
                    BaseField::from_u32_unchecked(c[k])
                }))
            })
            .collect(),
    )
}
