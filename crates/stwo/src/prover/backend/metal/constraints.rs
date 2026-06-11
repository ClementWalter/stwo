//! Apple-GPU constraint-quotient accumulation for AIRs that supply an MSL body.
//!
//! Constraint expressions are arbitrary Rust behind `EvalAtRow`, so they cannot run on
//! the GPU generically; instead an AIR can provide the body of a per-row evaluation in
//! Metal Shading Language, written against two macros:
//!
//! - `TRACE_AT(interaction, col)` — the column's value at the current row,
//! - `ADD_CONSTRAINT(v)` — accumulates `alpha^k * v` exactly like `EvalAtRow::add_constraint` (the
//!   constraint order must match the Rust evaluator).
//!
//! Only offset-0 mask reads are expressible in v1. The runner adds the accumulated
//! value times the coset-vanishing denominator inverse to the accumulation column,
//! mirroring the CPU finalize, so results are bit-identical to the CPU evaluator.
//! Trace columns are passed bindlessly (tier-2 argument buffers: a plain buffer of
//! `gpuAddress` values), which is how 100+ columns fit one dispatch.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::utils::uninit_vec;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;

/// Minimum rows for GPU dispatch.
pub const MIN_METAL_CONSTRAINT_LOG_SIZE: u32 = 16;

const KERNEL_TEMPLATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint P = 0x7FFFFFFFu;

inline uint m31_add(uint a, uint b) {
    uint s = a + b;
    return (s >= P) ? s - P : s;
}

inline uint m31_sub(uint a, uint b) {
    return (a >= b) ? a - b : a + P - b;
}

inline uint m31_mul(uint a, uint b) {
    ulong p = (ulong)a * (ulong)b;
    uint s = (uint)(p & P) + (uint)(p >> 31);
    s = (s & P) + (s >> 31);
    return (s >= P) ? s - P : s;
}

struct ColPtr { device const uint* data; };

struct Params {
    uint n_rows;
    uint trace_log_size;
    uint interaction_base[4];
};

kernel void accumulate_constraints(
    device const ColPtr* cols [[buffer(0)]],
    device const uint* alphas [[buffer(1)]],
    device const uint* denom_inv [[buffer(2)]],
    device const uint* acc_in0 [[buffer(3)]],
    device const uint* acc_in1 [[buffer(4)]],
    device const uint* acc_in2 [[buffer(5)]],
    device const uint* acc_in3 [[buffer(6)]],
    device uint* out0 [[buffer(7)]],
    device uint* out1 [[buffer(8)]],
    device uint* out2 [[buffer(9)]],
    device uint* out3 [[buffer(10)]],
    constant Params& p [[buffer(11)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= p.n_rows) { return; }
    uint acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    uint constraint_idx = 0;

#define TRACE_AT(interaction, col) (cols[p.interaction_base[(interaction)] + (col)].data[row])
#define ADD_CONSTRAINT(v) { \
    uint _c = (v); \
    acc0 = m31_add(acc0, m31_mul(_c, alphas[constraint_idx * 4 + 0])); \
    acc1 = m31_add(acc1, m31_mul(_c, alphas[constraint_idx * 4 + 1])); \
    acc2 = m31_add(acc2, m31_mul(_c, alphas[constraint_idx * 4 + 2])); \
    acc3 = m31_add(acc3, m31_mul(_c, alphas[constraint_idx * 4 + 3])); \
    constraint_idx++; \
}

//__AIR_BODY__

#undef TRACE_AT
#undef ADD_CONSTRAINT

    uint d = denom_inv[row >> p.trace_log_size];
    out0[row] = m31_add(acc_in0[row], m31_mul(acc0, d));
    out1[row] = m31_add(acc_in1[row], m31_mul(acc1, d));
    out2[row] = m31_add(acc_in2[row], m31_mul(acc2, d));
    out3[row] = m31_add(acc_in3[row], m31_mul(acc3, d));
}
"#;

struct ConstraintContext {
    device: Device,
    queue: CommandQueue,
    /// One compiled pipeline per distinct AIR body.
    pipelines: HashMap<String, ComputePipelineState>,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for ConstraintContext {}

fn context() -> Option<&'static Mutex<ConstraintContext>> {
    static CONTEXT: OnceLock<Option<Mutex<ConstraintContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let shared = super::context::gpu()?;
            let device = shared.device.clone();
            let queue = shared.queue.clone();
            Some(Mutex::new(ConstraintContext {
                device,
                queue,
                pipelines: HashMap::new(),
            }))
        })
        .as_ref()
}

#[repr(C)]
struct Params {
    n_rows: u32,
    trace_log_size: u32,
    interaction_base: [u32; 4],
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

/// Evaluates an AIR's constraints over all evaluation-domain rows on the GPU and
/// accumulates the quotients, given the AIR's MSL body (see the module docs for the
/// contract). `trace_columns` are grouped per interaction, in mask order. Returns
/// `None` when no usable device exists or the body fails to compile.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_constraints_metal(
    trace_columns: &[Vec<&[BaseField]>],
    random_coeff_powers: &[SecureField],
    denom_inv: &[BaseField],
    accum: &SecureColumnByCoords<CpuBackend>,
    n_rows: usize,
    trace_log_size: u32,
    air_body: &str,
) -> Option<SecureColumnByCoords<CpuBackend>> {
    assert!(trace_columns.len() <= 4);
    let ctx = context()?;
    let mut ctx = ctx.lock().unwrap();

    if !ctx.pipelines.contains_key(air_body) {
        let source = KERNEL_TEMPLATE.replace("//__AIR_BODY__", air_body);
        let library = ctx
            .device
            .new_library_with_source(&source, &CompileOptions::new())
            .inspect_err(|err| tracing::warn!("metal constraint body failed to compile: {err}"))
            .ok()?;
        let function = library.get_function("accumulate_constraints", None).ok()?;
        let pipeline = ctx
            .device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;
        ctx.pipelines.insert(air_body.to_string(), pipeline);
    }

    // Bind every trace column and collect their GPU addresses (bindless, tier 2).
    let mut interaction_base = [0u32; 4];
    let mut col_buffers = Vec::new();
    let mut addresses: Vec<u64> = Vec::new();
    for (interaction, columns) in trace_columns.iter().enumerate() {
        interaction_base[interaction] = addresses.len() as u32;
        for column in columns {
            let buffer = bind_input(&ctx.device, column);
            addresses.push(buffer.gpu_address());
            col_buffers.push(buffer);
        }
    }
    let addr_buffer = ctx.device.new_buffer_with_data(
        addresses.as_ptr() as *const std::ffi::c_void,
        (addresses.len() * 8).max(8) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let alphas: Vec<u32> = random_coeff_powers
        .iter()
        .flat_map(|alpha| alpha.to_m31_array().map(|v| v.0))
        .collect();
    let alpha_buffer = ctx.device.new_buffer_with_data(
        alphas.as_ptr() as *const std::ffi::c_void,
        (alphas.len() * 4).max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let denoms: Vec<u32> = denom_inv.iter().map(|v| v.0).collect();
    let denom_buffer = ctx.device.new_buffer_with_data(
        denoms.as_ptr() as *const std::ffi::c_void,
        (denoms.len() * 4).max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let acc_buffers: Vec<Buffer> = accum
        .columns
        .iter()
        .map(|c| bind_input(&ctx.device, c))
        .collect();
    // Safety: the kernel writes every entry before anything reads them.
    let mut out: [Vec<BaseField>; 4] = std::array::from_fn(|_| unsafe { uninit_vec(n_rows) });
    let out_buffers: Vec<Buffer> = out.iter().map(|c| bind_input(&ctx.device, c)).collect();

    let params = Params {
        n_rows: n_rows as u32,
        trace_log_size,
        interaction_base,
    };

    let pipeline = &ctx.pipelines[air_body];
    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&addr_buffer), 0);
    encoder.set_buffer(1, Some(&alpha_buffer), 0);
    encoder.set_buffer(2, Some(&denom_buffer), 0);
    for (k, buffer) in acc_buffers.iter().enumerate() {
        encoder.set_buffer(3 + k as u64, Some(buffer), 0);
    }
    for (k, buffer) in out_buffers.iter().enumerate() {
        encoder.set_buffer(7 + k as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        11,
        std::mem::size_of::<Params>() as u64,
        &params as *const _ as *const std::ffi::c_void,
    );
    // Bindless columns aren't tracked through the address buffer; declare residency.
    for buffer in &col_buffers {
        encoder.use_resource(buffer, MTLResourceUsage::Read);
    }
    encoder.dispatch_threads(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    for (vec, buffer) in out.iter_mut().zip(&out_buffers) {
        let zero_copy = std::ptr::eq(buffer.contents() as *const BaseField, vec.as_ptr());
        if !zero_copy {
            // Safety: the kernel wrote all entries into the shared buffer.
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
    Some(SecureColumnByCoords {
        columns: [c0, c1, c2, c3],
    })
}
