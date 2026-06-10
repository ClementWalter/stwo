//! Apple-GPU FRI folds.
//!
//! A fold pairs adjacent evaluations, applies an inverse butterfly with the inverted
//! domain coordinate, and combines `f0 + alpha * f1`. For both the line fold and the
//! circle-into-line fold, output row `i` uses the point `coset.at(bit_rev(i))` (the
//! even global index always lands in the half coset), with the line fold reading `x`
//! and the circle fold reading `y`. Threads compute their point by double-and-add and
//! the coordinate inverse by Fermat exponentiation, so values are bit-identical to the
//! CPU fold (group ops are exact; inverses are unique).

use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::utils::uninit_vec;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;

/// Minimum output size for GPU dispatch.
pub(crate) const MIN_METAL_FOLD_LOG_SIZE: u32 = 16;

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

// x^(P-2) = x^-1. P - 2 = 0b1111111111111111111111111111101.
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

struct CM31v { uint a; uint b; };

inline CM31v cm31_add(CM31v x, CM31v y) { return CM31v{m31_add(x.a, y.a), m31_add(x.b, y.b)}; }
inline CM31v cm31_sub(CM31v x, CM31v y) { return CM31v{m31_sub(x.a, y.a), m31_sub(x.b, y.b)}; }

inline CM31v cm31_mul(CM31v x, CM31v y) {
    return CM31v{
        m31_sub(m31_mul(x.a, y.a), m31_mul(x.b, y.b)),
        m31_add(m31_mul(x.a, y.b), m31_mul(x.b, y.a)),
    };
}

struct QM31v { CM31v a; CM31v b; };

// (a + b u)(c + d u) = (ac + (2+i) bd) + (ad + bc) u over CM31, u^2 = 2 + i.
inline QM31v qm31_mul(QM31v x, QM31v y) {
    CM31v ac = cm31_mul(x.a, y.a);
    CM31v bd = cm31_mul(x.b, y.b);
    CM31v r_bd = CM31v{
        m31_sub(m31_add(bd.a, bd.a), bd.b),
        m31_add(m31_add(bd.b, bd.b), bd.a),
    };
    CM31v lo = cm31_add(ac, r_bd);
    CM31v hi = cm31_add(cm31_mul(x.a, y.b), cm31_mul(x.b, y.a));
    return QM31v{lo, hi};
}

inline QM31v qm31_add(QM31v x, QM31v y) {
    return QM31v{cm31_add(x.a, y.a), cm31_add(x.b, y.b)};
}

inline QM31v qm31_sub(QM31v x, QM31v y) {
    return QM31v{cm31_sub(x.a, y.a), cm31_sub(x.b, y.b)};
}

inline QM31v qm31_scale_m31(QM31v x, uint m) {
    return QM31v{
        CM31v{m31_mul(x.a.a, m), m31_mul(x.a.b, m)},
        CM31v{m31_mul(x.b.a, m), m31_mul(x.b.b, m)},
    };
}

struct FoldParams {
    uint initial_x;
    uint initial_y;
    uint step_x;
    uint step_y;
    uint half_log;   // log2 of the output length
    uint coord_is_y; // 1: circle fold (y coordinate), 0: line fold (x coordinate)
    uint alpha[8];   // QM31 coordinates of alpha
};

kernel void fold(
    device const uint* in0 [[buffer(0)]],
    device const uint* in1 [[buffer(1)]],
    device const uint* in2 [[buffer(2)]],
    device const uint* in3 [[buffer(3)]],
    device uint* out0 [[buffer(4)]],
    device uint* out1 [[buffer(5)]],
    device uint* out2 [[buffer(6)]],
    device uint* out3 [[buffer(7)]],
    constant FoldParams& p [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= (1u << p.half_log)) { return; }
    uint j = (p.half_log == 0u) ? 0u : (reverse_bits(i) >> (32u - p.half_log));

    Point acc = Point{p.initial_x, p.initial_y};
    Point base = Point{p.step_x, p.step_y};
    uint k = j;
    while (k != 0u) {
        if (k & 1u) { acc = pt_add(acc, base); }
        base = pt_add(base, base);
        k >>= 1u;
    }
    uint coord_inv = m31_inv(p.coord_is_y ? acc.y : acc.x);

    uint e = i << 1;
    uint o = e + 1;
    QM31v f0 = QM31v{CM31v{in0[e], in1[e]}, CM31v{in2[e], in3[e]}};
    QM31v f1 = QM31v{CM31v{in0[o], in1[o]}, CM31v{in2[o], in3[o]}};
    // ibutterfly: (f0, f1) = (f0 + f1, (f0 - f1) * coord_inv).
    QM31v s = qm31_add(f0, f1);
    QM31v d = qm31_scale_m31(qm31_sub(f0, f1), coord_inv);
    QM31v alpha = QM31v{
        CM31v{p.alpha[0], p.alpha[1]},
        CM31v{p.alpha[2], p.alpha[3]},
    };
    QM31v r = qm31_add(qm31_mul(alpha, d), s);
    out0[i] = r.a.a;
    out1[i] = r.a.b;
    out2[i] = r.b.a;
    out3[i] = r.b.b;
}
"#;

struct FoldContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for FoldContext {}

fn context() -> Option<&'static Mutex<FoldContext>> {
    static CONTEXT: OnceLock<Option<Mutex<FoldContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let shared = super::context::gpu()?;
            let device = shared.device.clone();
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let function = library.get_function("fold", None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            let queue = shared.queue.clone();
            Some(Mutex::new(FoldContext {
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
struct FoldParams {
    initial_x: u32,
    initial_y: u32,
    step_x: u32,
    step_y: u32,
    half_log: u32,
    coord_is_y: u32,
    alpha: [u32; 8],
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

/// GPU FRI fold over secure-column coordinates. `coset` is the half coset whose points
/// supply the fold coordinate (`x` for line folds, `y` for circle folds). Returns
/// `None` without a usable device.
pub(crate) fn fold_metal(
    values: &SecureColumnByCoords<CpuBackend>,
    coset: Coset,
    coord_is_y: bool,
    alpha: SecureField,
) -> Option<SecureColumnByCoords<CpuBackend>> {
    let half_n = values.len() / 2;
    if half_n < (1 << MIN_METAL_FOLD_LOG_SIZE) {
        return None;
    }
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    // Safety: the kernel writes every entry before anything reads them.
    let mut out: [Vec<BaseField>; 4] = std::array::from_fn(|_| unsafe { uninit_vec(half_n) });
    let in_buffers: Vec<Buffer> = values
        .columns
        .iter()
        .map(|c| bind_input(&ctx.device, c))
        .collect();
    let out_buffers: Vec<Buffer> = out.iter().map(|c| bind_input(&ctx.device, c)).collect();

    let initial = coset.at(0);
    let alpha_coords = alpha.to_m31_array();
    let params = FoldParams {
        initial_x: initial.x.0,
        initial_y: initial.y.0,
        step_x: coset.step.x.0,
        step_y: coset.step.y.0,
        half_log: half_n.ilog2(),
        coord_is_y: u32::from(coord_is_y),
        alpha: [
            alpha_coords[0].0,
            alpha_coords[1].0,
            alpha_coords[2].0,
            alpha_coords[3].0,
            0,
            0,
            0,
            0,
        ],
    };
    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&ctx.pipeline);
    for (k, buffer) in in_buffers.iter().enumerate() {
        encoder.set_buffer(k as u64, Some(buffer), 0);
    }
    for (k, buffer) in out_buffers.iter().enumerate() {
        encoder.set_buffer(4 + k as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        8,
        std::mem::size_of::<FoldParams>() as u64,
        &params as *const _ as *const std::ffi::c_void,
    );
    encoder.dispatch_threads(MTLSize::new(half_n as u64, 1, 1), MTLSize::new(256, 1, 1));
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
                    half_n,
                );
            }
        }
    }
    let [c0, c1, c2, c3] = out;
    Some(SecureColumnByCoords {
        columns: [c0, c1, c2, c3],
    })
}

/// Chains all fold steps of one FRI layer and the packed Merkle tree of the final
/// folded evaluation into one submission. Returns the folded coordinates and, when the
/// folded size clears the tree threshold, the tree layers (leaves first, above-threshold
/// only). `None` without a usable device or for sub-threshold sizes.
#[allow(clippy::type_complexity)]
pub(crate) fn fold_line_chain_and_packed_tree_metal(
    values: &SecureColumnByCoords<CpuBackend>,
    mut coset: Coset,
    alphas: &[SecureField],
    is_m31_output: bool,
) -> Option<(
    SecureColumnByCoords<CpuBackend>,
    Option<Vec<Vec<crate::core::vcs::blake2_hash::Blake2sHash>>>,
)> {
    use crate::core::utils::uninit_vec;
    let n0 = values.len();
    if alphas.is_empty() || (n0 >> alphas.len()) < (1 << MIN_METAL_FOLD_LOG_SIZE) {
        return None;
    }
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    let command_buffer = ctx.queue.new_command_buffer();
    let mut in_buffers: Vec<Buffer> = values
        .columns
        .iter()
        .map(|c| bind_input(&ctx.device, c))
        .collect();
    // Keep every step's output storage alive until the wait.
    let mut step_outputs: Vec<[Vec<BaseField>; 4]> = Vec::with_capacity(alphas.len());
    let mut step_buffers: Vec<Vec<Buffer>> = Vec::with_capacity(alphas.len());
    for (step, &alpha) in alphas.iter().enumerate() {
        let half_n = n0 >> (step + 1);
        // Safety: the fold kernel writes every entry before anything reads them.
        let out: [Vec<BaseField>; 4] = std::array::from_fn(|_| unsafe { uninit_vec(half_n) });
        let out_buffers: Vec<Buffer> = out.iter().map(|c| bind_input(&ctx.device, c)).collect();

        let initial = coset.at(0);
        let alpha_coords = alpha.to_m31_array();
        let params = FoldParams {
            initial_x: initial.x.0,
            initial_y: initial.y.0,
            step_x: coset.step.x.0,
            step_y: coset.step.y.0,
            half_log: half_n.ilog2(),
            coord_is_y: 0,
            alpha: [
                alpha_coords[0].0,
                alpha_coords[1].0,
                alpha_coords[2].0,
                alpha_coords[3].0,
                0,
                0,
                0,
                0,
            ],
        };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        for (k, buffer) in in_buffers.iter().enumerate() {
            encoder.set_buffer(k as u64, Some(buffer), 0);
        }
        for (k, buffer) in out_buffers.iter().enumerate() {
            encoder.set_buffer(4 + k as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            8,
            std::mem::size_of::<FoldParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(MTLSize::new(half_n as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();

        in_buffers = out_buffers.clone();
        step_outputs.push(out);
        step_buffers.push(out_buffers);
        coset = coset.double();
    }

    // Packed tree over the final fold output, in the same submission.
    let final_half = n0 >> alphas.len();
    let n_leaves = final_half / 4;
    let pending = if n_leaves >= (1 << crate::prover::backend::metal::blake2s::MIN_METAL_LOG_SIZE) {
        super::blake2s::encode_packed_tree(
            command_buffer,
            step_buffers.last().unwrap(),
            n_leaves,
            is_m31_output,
        )
    } else {
        None
    };

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy out any fold output that couldn't bind zero-copy.
    let final_buffers = step_buffers.last().unwrap();
    let mut final_out = step_outputs.pop().unwrap();
    for (vec, buffer) in final_out.iter_mut().zip(final_buffers) {
        let zero_copy = std::ptr::eq(buffer.contents() as *const BaseField, vec.as_ptr());
        if !zero_copy {
            // Safety: the kernel wrote all entries into the shared buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const BaseField,
                    vec.as_mut_ptr(),
                    vec.len(),
                );
            }
        }
    }
    let [c0, c1, c2, c3] = final_out;
    Some((
        SecureColumnByCoords {
            columns: [c0, c1, c2, c3],
        },
        pending.map(super::blake2s::PendingTree::finish),
    ))
}
