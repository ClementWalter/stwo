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
use crate::core::fields::qm31::SecureField;
use crate::core::utils::uninit_vec;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;

/// Minimum rows for GPU dispatch.
pub(crate) const MIN_METAL_QUOTIENT_LOG_SIZE: u32 = 16;

const CHUNK_COLS: usize = 16;

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

struct AccumulateParams {
    uint n_rows;
    uint n_cols;     // valid columns in this chunk (1..=16)
    uint mode;       // bit0: first chunk (init from init_acc); bit1: last (emit coords)
    uint init_acc[4];
    uint coeffs[64]; // QM31 coordinates of c_i per column, [col][coord]
};

kernel void accumulate_numerators(
    device const uint* c0 [[buffer(0)]],
    device const uint* c1 [[buffer(1)]],
    device const uint* c2 [[buffer(2)]],
    device const uint* c3 [[buffer(3)]],
    device const uint* c4 [[buffer(4)]],
    device const uint* c5 [[buffer(5)]],
    device const uint* c6 [[buffer(6)]],
    device const uint* c7 [[buffer(7)]],
    device const uint* c8 [[buffer(8)]],
    device const uint* c9 [[buffer(9)]],
    device const uint* c10 [[buffer(10)]],
    device const uint* c11 [[buffer(11)]],
    device const uint* c12 [[buffer(12)]],
    device const uint* c13 [[buffer(13)]],
    device const uint* c14 [[buffer(14)]],
    device const uint* c15 [[buffer(15)]],
    device uint* state [[buffer(16)]],
    device uint* out0 [[buffer(17)]],
    device uint* out1 [[buffer(18)]],
    device uint* out2 [[buffer(19)]],
    device uint* out3 [[buffer(20)]],
    constant AccumulateParams& p [[buffer(21)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n_rows) { return; }
    device const uint* cols[16] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15};

    uint acc[4];
    if (p.mode & 1u) {
        for (uint k = 0; k < 4; k++) { acc[k] = p.init_acc[k]; }
    } else {
        for (uint k = 0; k < 4; k++) { acc[k] = state[i * 4 + k]; }
    }
    for (uint j = 0; j < p.n_cols; j++) {
        uint v = cols[j][i];
        for (uint k = 0; k < 4; k++) {
            acc[k] = m31_add(acc[k], m31_mul(v, p.coeffs[j * 4 + k]));
        }
    }
    if (p.mode & 2u) {
        out0[i] = acc[0];
        out1[i] = acc[1];
        out2[i] = acc[2];
        out3[i] = acc[3];
    } else {
        for (uint k = 0; k < 4; k++) { state[i * 4 + k] = acc[k]; }
    }
}
"#;

struct QuotientContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    combine_pipeline: Option<ComputePipelineState>,
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
            let device = Device::system_default()?;
            if !device.has_unified_memory() {
                return None;
            }
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let function = library.get_function("accumulate_numerators", None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            let queue = device.new_command_queue();
            Some(Mutex::new(QuotientContext {
                device,
                queue,
                pipeline,
                combine_pipeline: None,
                state_buffer: None,
            }))
        })
        .as_ref()
}

/// Forces device/pipeline initialization; see [`super::warmup`].
pub(crate) fn warmup() {
    let _ = context();
}

#[repr(C)]
struct AccumulateParams {
    n_rows: u32,
    n_cols: u32,
    mode: u32,
    init_acc: [u32; 4],
    coeffs: [u32; 64],
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

/// GPU partial-numerator accumulation over the first `n_rows` entries of each column:
/// `acc[row] = -b_sum + sum_i columns[i][row] * coeffs[i]`. Returns `None` when no
/// usable device exists.
pub(crate) fn accumulate_numerators_metal(
    columns: &[&[BaseField]],
    coeffs: &[SecureField],
    neg_b_sum: SecureField,
    n_rows: usize,
) -> Option<SecureColumnByCoords<CpuBackend>> {
    assert_eq!(columns.len(), coeffs.len());
    let ctx = context()?;
    let mut ctx = ctx.lock().unwrap();

    // Safety: the final GPU chunk writes every entry before anything reads them.
    let mut out: [Vec<BaseField>; 4] = std::array::from_fn(|_| unsafe { uninit_vec(n_rows) });

    let state_bytes = (n_rows * 16) as u64;
    if ctx
        .state_buffer
        .as_ref()
        .is_none_or(|b| b.length() < state_bytes)
    {
        ctx.state_buffer = Some(
            ctx.device
                .new_buffer(state_bytes, MTLResourceOptions::StorageModePrivate),
        );
    }

    let in_buffers: Vec<Buffer> = columns
        .iter()
        .map(|c| bind_input(&ctx.device, &c[..n_rows.min(c.len())]))
        .collect();
    let out_buffers: Vec<Buffer> = out.iter().map(|c| bind_input(&ctx.device, c)).collect();

    let command_buffer = ctx.queue.new_command_buffer();
    let n_chunks = columns.len().div_ceil(CHUNK_COLS);
    for (chunk_idx, (buf_chunk, coeff_chunk)) in in_buffers
        .chunks(CHUNK_COLS)
        .zip(coeffs.chunks(CHUNK_COLS))
        .enumerate()
    {
        let mut packed_coeffs = [0u32; 64];
        for (j, c) in coeff_chunk.iter().enumerate() {
            let arr = c.to_m31_array();
            for k in 0..4 {
                packed_coeffs[j * 4 + k] = arr[k].0;
            }
        }
        let params = AccumulateParams {
            n_rows: n_rows as u32,
            n_cols: buf_chunk.len() as u32,
            mode: u32::from(chunk_idx == 0) | (u32::from(chunk_idx == n_chunks - 1) << 1),
            init_acc: neg_b_sum.to_m31_array().map(|v| v.0),
            coeffs: packed_coeffs,
        };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        for slot in 0..CHUNK_COLS {
            // Unused slots must still be bound; alias the first column (never read).
            let buffer = buf_chunk.get(slot).unwrap_or(&buf_chunk[0]);
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_buffer(16, ctx.state_buffer.as_ref().map(|b| b as _), 0);
        for (k, buffer) in out_buffers.iter().enumerate() {
            encoder.set_buffer(17 + k as u64, Some(buffer), 0);
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
    command_buffer.wait_until_completed();

    // Copy out any column that couldn't bind zero-copy (small sizes).
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

const COMBINE_KERNEL: &str = r#"
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

struct CM31 { uint a; uint b; };

inline CM31 cm31_sub(CM31 x, CM31 y) { return CM31{m31_sub(x.a, y.a), m31_sub(x.b, y.b)}; }

inline CM31 cm31_mul(CM31 x, CM31 y) {
    return CM31{
        m31_sub(m31_mul(x.a, y.a), m31_mul(x.b, y.b)),
        m31_add(m31_mul(x.a, y.b), m31_mul(x.b, y.a)),
    };
}

// x^(P^2 - 2) = x^-1; the exponent 2^62 - 2^32 - 1 has all bits 0..61 set except 32.
inline CM31 cm31_inv(CM31 x) {
    CM31 r = x;
    for (int i = 60; i >= 0; i--) {
        r = cm31_mul(r, r);
        if (i != 32) { r = cm31_mul(r, x); }
    }
    return r;
}

struct Sample {
    // Re/Im of the sample point coordinates: prx, pix, pry, piy (each CM31).
    CM31 prx; CM31 pix; CM31 pry; CM31 piy;
    // first_linear_term_acc, QM31 as [a, b] CM31 pairs.
    CM31 flt0; CM31 flt1;
    // Lifting ratio of this accumulation: lifted_idx = (row >> (lr+1) << 1) | (row & 1).
    uint log_ratio;
    uint _pad;
};

struct Params {
    uint n_rows;
    uint n_samples;
};

kernel void combine_quotients(
    constant ulong* acc_addrs [[buffer(0)]],   // 4 coord columns per accumulation
    device const uint* xs [[buffer(1)]],
    device const uint* ys [[buffer(2)]],
    device uint* out0 [[buffer(3)]],
    device uint* out1 [[buffer(4)]],
    device uint* out2 [[buffer(5)]],
    device uint* out3 [[buffer(6)]],
    constant Sample* samples [[buffer(7)]],
    constant Params& p [[buffer(8)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= p.n_rows) { return; }
    uint x = xs[row];
    uint y = ys[row];

    CM31 q0 = CM31{0, 0};
    CM31 q1 = CM31{0, 0};
    for (uint s = 0; s < p.n_samples; s++) {
        constant Sample& sp = samples[s];
        // den = (prx - x) * piy - (pry - y) * pix, all CM31 with an M31 point.
        CM31 dx = CM31{m31_sub(sp.prx.a, x), sp.prx.b};
        CM31 dy = CM31{m31_sub(sp.pry.a, y), sp.pry.b};
        CM31 den = cm31_sub(cm31_mul(dx, sp.piy), cm31_mul(dy, sp.pix));
        CM31 inv = cm31_inv(den);

        uint lr = sp.log_ratio;
        uint lifted = ((row >> (lr + 1u)) << 1u) | (row & 1u);
        device const uint* a0 = (device const uint*)acc_addrs[s * 4 + 0];
        device const uint* a1 = (device const uint*)acc_addrs[s * 4 + 1];
        device const uint* a2 = (device const uint*)acc_addrs[s * 4 + 2];
        device const uint* a3 = (device const uint*)acc_addrs[s * 4 + 3];
        // full_numerator = acc.at(lifted) - first_linear_term * y (QM31, coordinate-wise
        // on the M31 y).
        CM31 n0 = CM31{
            m31_sub(a0[lifted], m31_mul(sp.flt0.a, y)),
            m31_sub(a1[lifted], m31_mul(sp.flt0.b, y)),
        };
        CM31 n1 = CM31{
            m31_sub(a2[lifted], m31_mul(sp.flt1.a, y)),
            m31_sub(a3[lifted], m31_mul(sp.flt1.b, y)),
        };
        // quotient += full_numerator.mul_cm31(inv).
        n0 = cm31_mul(n0, inv);
        n1 = cm31_mul(n1, inv);
        q0 = CM31{m31_add(q0.a, n0.a), m31_add(q0.b, n0.b)};
        q1 = CM31{m31_add(q1.a, n1.a), m31_add(q1.b, n1.b)};
    }
    out0[row] = q0.a;
    out1[row] = q0.b;
    out2[row] = q1.a;
    out3[row] = q1.b;
}
"#;

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

/// Returns the cached (x, y) coordinate columns of `domain` in bit-reversed order,
/// computing them once per (size, shift); they're domain constants shared by proofs.
fn domain_xy(domain: crate::core::poly::circle::CircleDomain) -> std::sync::Arc<XyColumns> {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::core::utils::bit_reverse_index;

    type Cache = HashMap<(u32, usize), Arc<XyColumns>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let key = (domain.log_size(), domain.half_coset.initial_index.0);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(columns) = cache.lock().unwrap().get(&key) {
        return columns.clone();
    }
    let log_size = domain.log_size();
    let n = domain.size();
    let mut xs = vec![BaseField::from_u32_unchecked(0); n];
    let mut ys = vec![BaseField::from_u32_unchecked(0); n];
    let fill = |(row, (x, y)): (usize, (&mut BaseField, &mut BaseField))| {
        let point = domain.at(bit_reverse_index(row, log_size));
        *x = point.x;
        *y = point.y;
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        xs.par_iter_mut()
            .zip(ys.par_iter_mut())
            .enumerate()
            .for_each(fill);
    }
    #[cfg(not(feature = "parallel"))]
    xs.iter_mut().zip(ys.iter_mut()).enumerate().for_each(fill);
    let columns = Arc::new(XyColumns { xs, ys });
    cache.lock().unwrap().entry(key).or_insert(columns).clone()
}

pub(crate) struct XyColumns {
    xs: Vec<BaseField>,
    ys: Vec<BaseField>,
}

/// GPU quotient combine over the subdomain: per row, the denominator inverses are
/// computed by Fermat exponentiation (the inverse is unique, so values equal
/// `CM31::batch_inverse` exactly) and every accumulation's lifted numerator is folded
/// in, mirroring the CPU loop bit for bit. Returns `None` without a usable device.
pub(crate) fn combine_quotients_metal(
    accumulations: &[crate::prover::pcs::quotient_ops::AccumulatedNumerators<CpuBackend>],
    subdomain: crate::core::poly::circle::CircleDomain,
    n_rows: usize,
) -> Option<SecureColumnByCoords<CpuBackend>> {
    use metal::MTLResourceUsage;
    if accumulations.is_empty() || accumulations.len() > 6 {
        return None;
    }
    let subdomain_log_size = subdomain.log_size();
    let xy = domain_xy(subdomain);
    let ctx = context()?;
    let mut ctx = ctx.lock().unwrap();
    if ctx.combine_pipeline.is_none() {
        let library = ctx
            .device
            .new_library_with_source(COMBINE_KERNEL, &CompileOptions::new())
            .ok()?;
        let function = library.get_function("combine_quotients", None).ok()?;
        ctx.combine_pipeline = Some(
            ctx.device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?,
        );
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
    let xs_buffer = bind_input(&ctx.device, &xy.xs[..n_rows]);
    let ys_buffer = bind_input(&ctx.device, &xy.ys[..n_rows]);

    // Safety: the kernel writes every entry before anything reads them.
    let mut out: [Vec<BaseField>; 4] = std::array::from_fn(|_| unsafe { uninit_vec(n_rows) });
    let out_buffers: Vec<Buffer> = out.iter().map(|c| bind_input(&ctx.device, c)).collect();

    let params = CombineParams {
        n_rows: n_rows as u32,
        n_samples: samples.len() as u32,
    };
    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(ctx.combine_pipeline.as_ref().unwrap());
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
    for buffer in &acc_buffers {
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
