//! Apple-GPU twiddle-tree precomputation.
//!
//! Layer `k` of the tree holds the x-coordinates of the `k`-times-doubled coset's
//! first-half points in bit-reversed order. Each GPU thread computes its point
//! directly as `initial + j * step` by double-and-add (the circle group is
//! associative, so values equal the CPU's sequential walk exactly), and inverse
//! twiddles by Fermat exponentiation (equal to batch inversion — inverses are
//! unique). The trailing padding element matches the CPU's constant.

use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::utils::uninit_vec;
use crate::prover::backend::CpuBackend;
use crate::prover::poly::twiddles::TwiddleTree;

/// Minimum coset size for GPU dispatch.
pub(crate) const MIN_METAL_TWIDDLE_LOG_SIZE: u32 = 16;

const KERNEL_SOURCE: &str = r#"
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

// x^(P-2) = x^-1. P - 2 = 2^31 - 3 = 0b1111111111111111111111111111101.
inline uint m31_inv(uint x) {
    uint r = x;
    for (int i = 29; i >= 0; i--) {
        r = m31_mul(r, r);
        if (i != 1) { r = m31_mul(r, x); }
    }
    return r;
}

struct Point { uint x; uint y; };

inline Point pt_add(Point a, Point b) {
    return Point{
        m31_sub(m31_mul(a.x, b.x), m31_mul(a.y, b.y)),
        m31_add(m31_mul(a.x, b.y), m31_mul(a.y, b.x)),
    };
}

struct LayerParams {
    uint initial_x;
    uint initial_y;
    uint step_x;
    uint step_y;
    uint layer_log; // log2 of the layer length (half the coset size)
    uint out_offset;
};

kernel void twiddle_layer(
    device uint* twiddles [[buffer(0)]],
    device uint* itwiddles [[buffer(1)]],
    constant LayerParams& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= (1u << p.layer_log)) { return; }
    // The layer is stored bit-reversed: output slot i holds point index rev(i).
    uint j = reverse_bits(i) >> (32u - p.layer_log);
    if (p.layer_log == 0u) { j = 0u; }

    // initial + j * step by double-and-add.
    Point acc = Point{p.initial_x, p.initial_y};
    Point base = Point{p.step_x, p.step_y};
    uint k = j;
    while (k != 0u) {
        if (k & 1u) { acc = pt_add(acc, base); }
        base = pt_add(base, base);
        k >>= 1u;
    }
    uint x = acc.x;
    twiddles[p.out_offset + i] = x;
    itwiddles[p.out_offset + i] = m31_inv(x);
}
"#;

struct TwiddleGenContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for TwiddleGenContext {}

fn context() -> Option<&'static Mutex<TwiddleGenContext>> {
    static CONTEXT: OnceLock<Option<Mutex<TwiddleGenContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let shared = super::context::gpu()?;
            let device = shared.device.clone();
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let function = library.get_function("twiddle_layer", None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            let queue = shared.queue.clone();
            Some(Mutex::new(TwiddleGenContext {
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

#[repr(C)]
struct LayerParams {
    initial_x: u32,
    initial_y: u32,
    step_x: u32,
    step_y: u32,
    layer_log: u32,
    out_offset: u32,
}

fn bind_output(device: &Device, data: &mut [BaseField]) -> Option<Buffer> {
    let bytes = std::mem::size_of_val(data);
    let page = 16384;
    ((data.as_ptr() as usize).is_multiple_of(page) && bytes.is_multiple_of(page)).then(|| {
        device.new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const std::ffi::c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    })
}

/// GPU twiddle-tree precomputation; values are bit-identical to the CPU path. Returns
/// `None` without a usable device or zero-copy-bindable buffers.
pub(crate) fn precompute_twiddles_metal(mut coset: Coset) -> Option<TwiddleTree<CpuBackend>> {
    if coset.log_size() < MIN_METAL_TWIDDLE_LOG_SIZE {
        return None;
    }
    let root_coset = coset;
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    let total = coset.size();
    // Safety: every entry below `total - 1` is written by a kernel; the final padding
    // slot is written on the CPU below.
    let mut twiddles: Vec<BaseField> = unsafe { uninit_vec(total) };
    let mut itwiddles: Vec<BaseField> = unsafe { uninit_vec(total) };
    let (tw_buffer, itw_buffer) = (
        bind_output(&ctx.device, &mut twiddles)?,
        bind_output(&ctx.device, &mut itwiddles)?,
    );

    let command_buffer = ctx.queue.new_command_buffer();
    let mut offset = 0u32;
    for _ in 0..coset.log_size() {
        let half_log = coset.log_size() - 1;
        let initial = coset.at(0);
        let step = coset.step;
        let params = LayerParams {
            initial_x: initial.x.0,
            initial_y: initial.y.0,
            step_x: step.x.0,
            step_y: step.y.0,
            layer_log: half_log,
            out_offset: offset,
        };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        encoder.set_buffer(0, Some(&tw_buffer), 0);
        encoder.set_buffer(1, Some(&itw_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<LayerParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(MTLSize::new(1 << half_log, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        offset += 1 << half_log;
        coset = coset.double();
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // The CPU path pads the buffer to a power of two with this constant.
    twiddles[total - 1] = BaseField::from_u32_unchecked(1);
    itwiddles[total - 1] = BaseField::from_u32_unchecked(1);

    Some(TwiddleTree {
        root_coset,
        twiddles,
        itwiddles,
    })
}
