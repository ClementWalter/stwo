//! Apple-GPU circle FFT kernels.
//!
//! A transform is run as a small number of tiled passes: each threadgroup stages a
//! 2^TILE_LOG-element tile in threadgroup memory and applies up to TILE_LOG butterfly
//! layers locally (barriers between layers), so a 2^22 transform needs two memory
//! passes instead of one per layer. A pass at base layer `i0` views the array as
//! `[hi | r (a bits) | mid | c (c_log bits)]` and butterflies along `r` (stride
//! 2^i0); `c` keeps global accesses in contiguous runs. Twiddle layers are packed
//! into one GPU buffer per (root coset, domain, direction) and cached.
//!
//! Layer order, butterfly formulas and twiddle indexing mirror the scalar reference
//! (`interpolate_scalar` / `evaluate_into_scalar`) exactly; outputs are bit-identical
//! and reference tests pin them.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::prover::backend::cpu::circle::circle_twiddles_from_line_twiddles;
use crate::prover::backend::CpuBackend;
use crate::prover::poly::circle::EvalsOrCoeffs;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::Poly;

/// log2 of the tile staged in threadgroup memory (2^13 u32 = 32KB).
const TILE_LOG: u32 = 13;
const THREADS_PER_GROUP: u64 = 1024;
/// Minimum transform size for GPU dispatch.
pub(crate) const MIN_METAL_FFT_LOG_SIZE: u32 = 14;

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

struct PassParams {
    uint i0;        // global layer index of the pass's first layer
    uint n_layers;  // butterfly layers in this pass (<= TILE_LOG)
    uint c_log;     // contiguous low bits kept inside the tile
    uint inverse;   // 1: ibutterfly (ifft), 0: butterfly (rfft)
    uint descending;// 1: apply the pass's layers from highest to lowest (rfft)
    uint scale;     // multiply outputs by this on the final store (1 = no-op)
    // When nonzero, stage from `src` (zero-extended past src_len) instead of `values`,
    // fusing the coefficient extension into the transform's first pass.
    uint src_len;
    // Offset of each layer's twiddle slice in the packed twiddle buffer, indexed by
    // LOCAL layer j (0..n_layers).
    uint twiddle_offsets[16];
};

constant uint TILE_LOG = 13;

kernel void fft_pass(
    device uint* values [[buffer(0)]],
    device const uint* twiddles [[buffer(1)]],
    constant PassParams& p [[buffer(2)]],
    device const uint* src [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup uint tile[1 << TILE_LOG];

    const uint a = p.n_layers;
    const uint c_log = p.c_log;
    const uint mid_bits = p.i0 - c_log;
    const uint g_hi = gid >> mid_bits;
    const uint g_mid = gid & ((1u << mid_bits) - 1u);
    const uint base = (g_hi << (p.i0 + a)) + (g_mid << c_log);
    const uint span_log = a + c_log; // == TILE_LOG for full tiles
    const uint n_local = 1u << span_log;
    const uint c_mask = (1u << c_log) - 1u;

    // Stage the tile: local index = [r | c], global = base + r*2^i0 + c.
    for (uint local = tid; local < n_local; local += 1024u) {
        uint r = local >> c_log;
        uint c = local & c_mask;
        uint g = base + (r << p.i0) + c;
        tile[local] = (p.src_len != 0u) ? ((g < p.src_len) ? src[g] : 0u) : values[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < a; step++) {
        uint j = p.descending ? (a - 1u - step) : step;
        device const uint* tw = twiddles + p.twiddle_offsets[j];
        uint half_r = 1u << j;
        // Pairs: r with bit j clear vs set; iterate pair index q over r-pairs x c.
        uint n_pairs = n_local >> 1;
        for (uint q = tid; q < n_pairs; q += 1024u) {
            uint rq = q >> c_log;          // pair index in the r dimension
            uint c = q & c_mask;
            uint r0 = ((rq >> j) << (j + 1)) | (rq & (half_r - 1u));
            uint l0 = (r0 << c_log) | c;
            uint l1 = ((r0 + half_r) << c_log) | c;
            // h_global = [g_hi | r0 >> (j+1)]
            uint t = tw[(g_hi << (a - 1u - j)) | (rq >> j)];
            uint v0 = tile[l0];
            uint v1 = tile[l1];
            if (p.inverse) {
                tile[l0] = m31_add(v0, v1);
                tile[l1] = m31_mul(m31_sub(v0, v1), t);
            } else {
                uint tmp = m31_mul(v1, t);
                tile[l0] = m31_add(v0, tmp);
                tile[l1] = m31_sub(v0, tmp);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint local = tid; local < n_local; local += 1024u) {
        uint r = local >> c_log;
        uint c = local & c_mask;
        uint v = tile[local];
        if (p.scale != 1u) { v = m31_mul(v, p.scale); }
        values[base + (r << p.i0) + c] = v;
    }
}
"#;

struct FftContext {
    device: Device,
    queue: CommandQueue,
    pass_pipeline: ComputePipelineState,
    /// Packed twiddle buffers keyed by
    /// (root initial index, root log size, domain log size, inverse).
    twiddle_cache: HashMap<(u32, u32, u32, bool), TwiddleBuffers>,
}

struct TwiddleBuffers {
    buffer: Buffer,
    /// Offset of layer `L`'s twiddle slice (L = 0 is the circle layer).
    layer_offsets: Vec<u32>,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for FftContext {}

fn context() -> Option<&'static Mutex<FftContext>> {
    static CONTEXT: OnceLock<Option<Mutex<FftContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let shared = super::context::gpu()?;
            let device = shared.device.clone();
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let pass = library.get_function("fft_pass", None).ok()?;
            let pass_pipeline = device
                .new_compute_pipeline_state_with_function(&pass)
                .ok()?;
            let queue = shared.queue.clone();
            Some(Mutex::new(FftContext {
                device,
                queue,
                pass_pipeline,
                twiddle_cache: HashMap::new(),
            }))
        })
        .as_ref()
}

/// Forces device/pipeline initialization; see [`super::warmup`].
pub(crate) fn warmup() {
    let _ = context();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PassParams {
    i0: u32,
    n_layers: u32,
    c_log: u32,
    inverse: u32,
    descending: u32,
    scale: u32,
    src_len: u32,
    twiddle_offsets: [u32; 16],
}

/// Packs the domain's twiddle layers (circle layer first, then line layers) into one
/// GPU buffer, mirroring the scalar reference's layer order and indexing.
fn pack_twiddles(
    ctx: &mut FftContext,
    domain: CircleDomain,
    root_coset: Coset,
    tree_buffer: &[BaseField],
    inverse: bool,
) -> (*const TwiddleBuffers, u32) {
    let domain_log = domain.log_size();
    let key = (
        root_coset.initial_index.0 as u32,
        root_coset.log_size,
        domain_log,
        inverse,
    );
    if !ctx.twiddle_cache.contains_key(&key) {
        let line_twiddles = domain_line_twiddles_from_tree(domain, tree_buffer);
        let circle_twiddles: Vec<BaseField> =
            circle_twiddles_from_line_twiddles(line_twiddles[0]).collect();
        let mut packed: Vec<u32> = Vec::with_capacity(domain.size());
        let mut layer_offsets = Vec::with_capacity(domain_log as usize);
        layer_offsets.push(0);
        packed.extend(circle_twiddles.iter().map(|t| t.0));
        for layer in &line_twiddles {
            layer_offsets.push(packed.len() as u32);
            packed.extend(layer.iter().map(|t| t.0));
        }
        let buffer = ctx.device.new_buffer_with_data(
            packed.as_ptr() as *const std::ffi::c_void,
            (packed.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        ctx.twiddle_cache.insert(
            key,
            TwiddleBuffers {
                buffer,
                layer_offsets,
            },
        );
    }
    (&ctx.twiddle_cache[&key] as *const _, domain_log)
}

/// Splits `n_log` layers into passes. The first pass is contiguous (`c_log = 0`) and
/// takes a full tile of layers; later passes are strided and capped so the tile keeps
/// at least 2^5 contiguous lanes (128-byte global-memory runs stay coalesced).
fn pass_split(n_log: u32) -> Vec<(u32, u32)> {
    const MIN_C_LOG: u32 = 5;
    // (i0, n_layers) per pass, in ascending layer order.
    let mut passes = vec![];
    let mut i0 = 0;
    while i0 < n_log {
        let cap = if i0 == 0 {
            TILE_LOG
        } else {
            (TILE_LOG - MIN_C_LOG).min(i0)
        };
        let n_layers = (n_log - i0).min(cap);
        passes.push((i0, n_layers));
        i0 += n_layers;
    }
    passes
}

#[allow(clippy::too_many_arguments)]
fn encode_pass(
    ctx: &FftContext,
    command_buffer: &metal::CommandBufferRef,
    values: &Buffer,
    twiddles: &TwiddleBuffers,
    n_log: u32,
    (i0, n_layers): (u32, u32),
    inverse: bool,
    descending: bool,
    scale: u32,
    src: Option<(&Buffer, u32)>,
) {
    let c_log = (TILE_LOG - n_layers).min(i0);
    debug_assert!(n_layers <= 16);
    let mut offsets = [0u32; 16];
    for j in 0..n_layers {
        offsets[j as usize] = twiddles.layer_offsets[(i0 + j) as usize];
    }
    let params = PassParams {
        i0,
        n_layers,
        c_log,
        inverse: u32::from(inverse),
        descending: u32::from(descending),
        scale,
        src_len: src.map_or(0, |(_, len)| len),
        twiddle_offsets: offsets,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&ctx.pass_pipeline);
    encoder.set_buffer(0, Some(values), 0);
    encoder.set_buffer(1, Some(&twiddles.buffer), 0);
    encoder.set_buffer(3, Some(src.map_or(values, |(buffer, _)| buffer)), 0);
    encoder.set_bytes(
        2,
        std::mem::size_of::<PassParams>() as u64,
        &params as *const _ as *const std::ffi::c_void,
    );
    let n_groups = 1u64 << (n_log - n_layers - c_log);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_groups, 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
    encoder.end_encoding();
}

/// Binds a page-aligned, page-multiple slice zero-copy; returns `None` otherwise.
fn bind_zero_copy(device: &Device, data: &[BaseField]) -> Option<Buffer> {
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

/// Batched in-place GPU inverse circle FFTs: every column's passes encoded into one
/// command buffer, one synchronization. Returns the evaluations on `Err` when the GPU
/// path can't take them.
#[allow(clippy::type_complexity)]
pub(crate) fn ifft_batch_metal(
    columns: Vec<
        crate::prover::poly::circle::CircleEvaluation<
            CpuBackend,
            BaseField,
            crate::prover::poly::BitReversedOrder,
        >,
    >,
    twiddles: &TwiddleTree<CpuBackend>,
) -> Result<
    Vec<crate::prover::poly::circle::CircleCoefficients<CpuBackend>>,
    Vec<
        crate::prover::poly::circle::CircleEvaluation<
            CpuBackend,
            BaseField,
            crate::prover::poly::BitReversedOrder,
        >,
    >,
> {
    if columns
        .iter()
        .any(|eval| eval.domain.log_size() < MIN_METAL_FFT_LOG_SIZE)
    {
        return Err(columns);
    }
    let Some(ctx) = context() else {
        return Err(columns);
    };
    let mut ctx = ctx.lock().unwrap();

    let mut work: Vec<(Vec<BaseField>, CircleDomain)> = columns
        .into_iter()
        .map(|eval| (eval.values, eval.domain))
        .collect();
    let mut bindings = Vec::with_capacity(work.len());
    for (values, _) in &work {
        let Some(buffer) = bind_zero_copy(&ctx.device, values) else {
            return Err(repack_evals(work));
        };
        bindings.push(buffer);
    }
    let domains: Vec<CircleDomain> = work.iter().map(|w| w.1).collect();
    for &domain in &domains {
        pack_twiddles(
            &mut ctx,
            domain,
            twiddles.root_coset,
            &twiddles.itwiddles,
            true,
        );
    }

    let command_buffer = ctx.queue.new_command_buffer();
    for ((_, domain), buffer) in work.iter().zip(&bindings) {
        let n_log = domain.log_size();
        let key = (
            twiddles.root_coset.initial_index.0 as u32,
            twiddles.root_coset.log_size,
            n_log,
            true,
        );
        let tw = &ctx.twiddle_cache[&key];
        let n_inv = BaseField::from_u32_unchecked(domain.size() as u32)
            .inverse()
            .0;
        let passes = pass_split(n_log);
        let last = passes.len() - 1;
        for (idx, &pass) in passes.iter().enumerate() {
            let scale = if idx == last { n_inv } else { 1 };
            encode_pass(
                &ctx,
                command_buffer,
                buffer,
                tw,
                n_log,
                pass,
                true,
                false,
                scale,
                None,
            );
        }
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(work
        .drain(..)
        .map(|(values, _)| crate::prover::poly::circle::CircleCoefficients::new(values))
        .collect())
}

fn repack_evals(
    work: Vec<(Vec<BaseField>, CircleDomain)>,
) -> Vec<
    crate::prover::poly::circle::CircleEvaluation<
        CpuBackend,
        BaseField,
        crate::prover::poly::BitReversedOrder,
    >,
> {
    work.into_iter()
        .map(|(values, domain)| crate::prover::poly::circle::CircleEvaluation::new(domain, values))
        .collect()
}

/// In-place GPU inverse circle FFT over `values` (bit-reversed circle-domain order),
/// leaving natural-order FFT-basis coefficients scaled by `1/N` — exactly
/// [`interpolate_scalar`]'s output. Returns `false` (values untouched) when no usable
/// device exists or the buffer isn't zero-copy bindable.
///
/// [`interpolate_scalar`]: crate::prover::backend::cpu::circle::interpolate_scalar
pub(crate) fn ifft_metal(
    values: &mut [BaseField],
    domain: CircleDomain,
    twiddles: &TwiddleTree<CpuBackend>,
) -> bool {
    let n_log = domain.log_size();
    if n_log < MIN_METAL_FFT_LOG_SIZE {
        return false;
    }
    let Some(ctx) = context() else {
        return false;
    };
    let mut ctx = ctx.lock().unwrap();
    let Some(values_buffer) = bind_zero_copy(&ctx.device, values) else {
        return false;
    };

    let (tw_ptr, _) = pack_twiddles(
        &mut ctx,
        domain,
        twiddles.root_coset,
        &twiddles.itwiddles,
        true,
    );
    // Safety: the cache entry lives as long as the context; no eviction.
    let tw = unsafe { &*tw_ptr };

    let n_inv = BaseField::from_u32_unchecked(domain.size() as u32)
        .inverse()
        .0;
    let passes = pass_split(n_log);
    let last = passes.len() - 1;
    let command_buffer = ctx.queue.new_command_buffer();
    for (idx, &pass) in passes.iter().enumerate() {
        let scale = if idx == last { n_inv } else { 1 };
        encode_pass(
            &ctx,
            command_buffer,
            &values_buffer,
            tw,
            n_log,
            pass,
            true,
            false,
            scale,
            None,
        );
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();
    true
}

/// GPU circle FFT from `coeffs` (natural order) onto `domain`, writing evaluations to
/// `out` — exactly [`evaluate_into_scalar`]'s output. Returns `false` (out untouched)
/// when no usable device exists or the buffers aren't zero-copy bindable.
///
/// [`evaluate_into_scalar`]: crate::prover::backend::cpu::circle::evaluate_into_scalar
pub(crate) fn rfft_metal(
    coeffs: &[BaseField],
    domain: CircleDomain,
    twiddles: &TwiddleTree<CpuBackend>,
    out: &mut [BaseField],
) -> bool {
    let n_log = domain.log_size();
    if n_log < MIN_METAL_FFT_LOG_SIZE || coeffs.len() < 2 {
        return false;
    }
    assert_eq!(out.len(), domain.size());
    let Some(ctx) = context() else {
        return false;
    };
    let mut ctx = ctx.lock().unwrap();
    let (Some(out_buffer), Some(coeffs_buffer)) = (
        bind_zero_copy(&ctx.device, out),
        bind_zero_copy(&ctx.device, coeffs),
    ) else {
        return false;
    };

    let (tw_ptr, _) = pack_twiddles(
        &mut ctx,
        domain,
        twiddles.root_coset,
        &twiddles.twiddles,
        false,
    );
    // Safety: the cache entry lives as long as the context; no eviction.
    let tw = unsafe { &*tw_ptr };

    let command_buffer = ctx.queue.new_command_buffer();
    // Layers run from highest to lowest: passes in reverse, descending within a pass.
    // The first pass stages from the coefficient buffer (zero-extended), fusing the
    // extension into the transform.
    for (idx, &pass) in pass_split(n_log).iter().rev().enumerate() {
        let src = (idx == 0).then_some((&coeffs_buffer, coeffs.len() as u32));
        encode_pass(
            &ctx,
            command_buffer,
            &out_buffer,
            tw,
            n_log,
            pass,
            false,
            true,
            1,
            src,
        );
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();
    true
}

/// Batched fused interpolate+extend for the commit path: encodes every column's ifft
/// and rfft passes into one command buffer and waits once, so GPU work streams
/// back-to-back instead of synchronizing per column. Returns the columns on `Err`
/// when the GPU path can't take them (no device, small or unalignable columns).
#[allow(clippy::type_complexity)]
pub(crate) fn fused_transform_metal(
    columns: Vec<EvalsOrCoeffs<CpuBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CpuBackend>,
    store_polynomials_coefficients: bool,
) -> Result<Vec<Poly<CpuBackend>>, Vec<EvalsOrCoeffs<CpuBackend>>> {
    fused_transform_metal_with_itwiddles(
        columns,
        log_blowup_factor,
        twiddles,
        None,
        store_polynomials_coefficients,
    )
}

/// Like [`fused_transform_metal`], with a separate inverse-twiddle tree for the
/// interpolation step — needed when the evaluations live on a split subdomain, whose
/// twiddles are a strided extraction from the full tree rather than its tail layers.
#[allow(clippy::type_complexity)]
pub(crate) fn fused_transform_metal_with_itwiddles(
    columns: Vec<EvalsOrCoeffs<CpuBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CpuBackend>,
    ifft_twiddles: Option<&TwiddleTree<CpuBackend>>,
    store_polynomials_coefficients: bool,
) -> Result<Vec<Poly<CpuBackend>>, Vec<EvalsOrCoeffs<CpuBackend>>> {
    fused_transform_metal_chained(
        columns,
        log_blowup_factor,
        twiddles,
        ifft_twiddles,
        store_polynomials_coefficients,
        |_, _| Some(()),
    )
    .map(|(polys, _)| polys)
}

/// Like the fused transform, additionally letting `chain` encode follow-up kernels
/// (e.g. the Merkle tree over these evaluations) into the same submission, right after
/// the extension transforms and before the single wait. `chain` receives the command
/// buffer and the per-column LDE output buffers; a `None` from it only omits the
/// chained outputs — the transforms still complete.
#[allow(clippy::type_complexity)]
pub(crate) fn fused_transform_metal_chained<R>(
    columns: Vec<EvalsOrCoeffs<CpuBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CpuBackend>,
    ifft_twiddles: Option<&TwiddleTree<CpuBackend>>,
    store_polynomials_coefficients: bool,
    chain: impl FnOnce(&metal::CommandBufferRef, &[Buffer]) -> Option<R>,
) -> Result<(Vec<Poly<CpuBackend>>, Option<R>), Vec<EvalsOrCoeffs<CpuBackend>>> {
    let itw = ifft_twiddles.unwrap_or(twiddles);
    let small = columns.iter().any(|column| {
        let log_size = match column {
            EvalsOrCoeffs::Evals(evals) => evals.domain.log_size(),
            EvalsOrCoeffs::Coeffs(coeffs) => coeffs.log_size(),
        };
        log_size < MIN_METAL_FFT_LOG_SIZE
    });
    if small {
        return Err(columns);
    }
    let Some(ctx) = context() else {
        return Err(columns);
    };
    let mut ctx = ctx.lock().unwrap();

    // Unpack: (values, domain, needs_ifft) per column, plus a zeroed output vector.
    let t_alloc = std::time::Instant::now();
    let mut work: Vec<(Vec<BaseField>, CircleDomain, bool, Vec<BaseField>)> = columns
        .into_iter()
        .map(|column| {
            let (values, domain, needs_ifft) = match column {
                EvalsOrCoeffs::Evals(evals) => (evals.values, evals.domain, true),
                EvalsOrCoeffs::Coeffs(coeffs) => {
                    let domain = CanonicCoset::new(coeffs.log_size()).circle_domain();
                    (coeffs.coeffs, domain, false)
                }
            };
            let ext_size = 1usize << (domain.log_size() + log_blowup_factor);
            // Safety: the extend kernel writes every element before anything reads it.
            let out: Vec<BaseField> = unsafe { crate::core::utils::uninit_vec(ext_size) };
            (values, domain, needs_ifft, out)
        })
        .collect();

    // Bind everything zero-copy up front; bail out (returning ownership) on failure.
    let mut bindings = Vec::with_capacity(work.len());
    for (values, _, _, out) in &work {
        let Some(values_buffer) = bind_zero_copy(&ctx.device, values) else {
            return Err(repack(work));
        };
        let Some(out_buffer) = bind_zero_copy(&ctx.device, out) else {
            return Err(repack(work));
        };
        bindings.push((values_buffer, out_buffer));
    }

    // Columns can have distinct sizes; ensure every (domain, direction) is cached
    // before encoding (cache insertion needs &mut ctx).
    let domains: Vec<CircleDomain> = work.iter().map(|w| w.1).collect();
    for &domain in &domains {
        let ext_domain = CanonicCoset::new(domain.log_size() + log_blowup_factor).circle_domain();
        pack_twiddles(&mut ctx, domain, itw.root_coset, &itw.itwiddles, true);
        pack_twiddles(
            &mut ctx,
            ext_domain,
            twiddles.root_coset,
            &twiddles.twiddles,
            false,
        );
    }

    tracing::debug!("metal fused: alloc+bind {:?}", t_alloc.elapsed());
    let t_encode = std::time::Instant::now();

    // First command buffer: every column's ifft, in place.
    let ifft_buffer = ctx.queue.new_command_buffer();
    for ((_, domain, needs_ifft, _), (values_buffer, _)) in work.iter().zip(&bindings) {
        if !*needs_ifft {
            continue;
        }
        let n_log = domain.log_size();
        let key = (
            itw.root_coset.initial_index.0 as u32,
            itw.root_coset.log_size,
            n_log,
            true,
        );
        let tw = &ctx.twiddle_cache[&key];
        let n_inv = BaseField::from_u32_unchecked(domain.size() as u32)
            .inverse()
            .0;
        let passes = pass_split(n_log);
        let last = passes.len() - 1;
        for (idx, &pass) in passes.iter().enumerate() {
            let scale = if idx == last { n_inv } else { 1 };
            encode_pass(
                &ctx,
                ifft_buffer,
                values_buffer,
                tw,
                n_log,
                pass,
                true,
                false,
                scale,
                None,
            );
        }
    }
    ifft_buffer.commit();

    // While the GPU interpolates, fault the fresh output pages in on the CPU so the
    // extension kernels don't stall on first-touch page faults.
    {
        let page_elems = 16384 / 4;
        let prefault = |out: &mut [BaseField]| {
            for slot in out.iter_mut().step_by(page_elems) {
                *slot = BaseField::from_u32_unchecked(0);
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            work.par_iter_mut().for_each(|(_, _, _, out)| prefault(out));
        }
        #[cfg(not(feature = "parallel"))]
        work.iter_mut().for_each(|(_, _, _, out)| prefault(out));
    }
    ifft_buffer.wait_until_completed();

    let command_buffer = ctx.queue.new_command_buffer();
    for ((values, domain, _, out), (values_buffer, out_buffer)) in work.iter().zip(&bindings) {
        let n_log = domain.log_size();
        let ext_domain = CanonicCoset::new(n_log + log_blowup_factor).circle_domain();
        let _ = out;
        let ext_log = ext_domain.log_size();
        let key = (
            twiddles.root_coset.initial_index.0 as u32,
            twiddles.root_coset.log_size,
            ext_log,
            false,
        );
        let tw = &ctx.twiddle_cache[&key];
        // The first (highest) pass stages from the coefficient buffer, zero-extended,
        // fusing the extension into the transform.
        for (idx, &pass) in pass_split(ext_log).iter().rev().enumerate() {
            let src = (idx == 0).then_some((values_buffer, values.len() as u32));
            encode_pass(
                &ctx,
                command_buffer,
                out_buffer,
                tw,
                ext_log,
                pass,
                false,
                true,
                1,
                src,
            );
        }
    }
    let out_buffers: Vec<Buffer> = bindings.iter().map(|(_, out)| out.clone()).collect();
    let chained = chain(command_buffer, &out_buffers);
    tracing::debug!("metal fused: encode {:?}", t_encode.elapsed());
    let t_wait = std::time::Instant::now();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    tracing::debug!("metal fused: gpu {:?}", t_wait.elapsed());

    let polys = work
        .into_iter()
        .map(|(values, domain, _, out)| {
            let ext_domain =
                CanonicCoset::new(domain.log_size() + log_blowup_factor).circle_domain();
            let coeffs = store_polynomials_coefficients
                .then(|| crate::prover::poly::circle::CircleCoefficients::new(values));
            Poly::new(
                coeffs,
                crate::prover::poly::circle::CircleEvaluation::new(ext_domain, out),
            )
        })
        .collect();
    Ok((polys, chained))
}

fn repack(
    work: Vec<(Vec<BaseField>, CircleDomain, bool, Vec<BaseField>)>,
) -> Vec<EvalsOrCoeffs<CpuBackend>> {
    work.into_iter()
        .map(|(values, domain, needs_ifft, _)| {
            if needs_ifft {
                EvalsOrCoeffs::Evals(crate::prover::poly::circle::CircleEvaluation::new(
                    domain, values,
                ))
            } else {
                EvalsOrCoeffs::Coeffs(crate::prover::poly::circle::CircleCoefficients::new(values))
            }
        })
        .collect()
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::cpu::CpuBackend;
    use crate::prover::poly::circle::PolyOps;

    /// GPU kernel micro-benchmark (CPU-load insensitive). Run manually:
    /// `cargo test -p stwo --features prover,metal --release --lib fft_kernel_bench -- --ignored
    /// --nocapture`
    #[test]
    #[ignore]
    fn fft_kernel_bench() {
        const LOG: u32 = 22;
        let twiddles = <CpuBackend as PolyOps>::precompute_twiddles(
            CanonicCoset::new(LOG + 2).circle_domain().half_coset,
        );
        let domain = CanonicCoset::new(LOG).circle_domain();
        let ext_domain = CanonicCoset::new(LOG + 1).circle_domain();
        let mut values: Vec<BaseField> = (0..1u32 << LOG)
            .map(|i| BaseField::from_u32_unchecked((i.wrapping_mul(2654435761)) >> 1))
            .collect();
        let mut out = vec![BaseField::from_u32_unchecked(0); 1 << (LOG + 1)];
        // Warm: twiddle upload + pipeline.
        assert!(ifft_metal(&mut values, domain, &twiddles));
        assert!(rfft_metal(&values, ext_domain, &twiddles, &mut out));
        let n = 20;
        let t = std::time::Instant::now();
        for _ in 0..n {
            ifft_metal(&mut values, domain, &twiddles);
        }
        let ifft_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let t = std::time::Instant::now();
        for _ in 0..n {
            rfft_metal(&values, ext_domain, &twiddles, &mut out);
        }
        let rfft_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        std::println!(
            "ifft@2^{LOG}: {ifft_ms:.2} ms ({:.0} GB/s), rfft->2^{}: {rfft_ms:.2} ms ({:.0} GB/s)",
            (2.0 * 4.0 * (1u64 << LOG) as f64 * 2.0) / ifft_ms / 1e6,
            LOG + 1,
            (3.0 * 4.0 * (1u64 << (LOG + 1)) as f64 * 2.0) / rfft_ms / 1e6,
        );
    }
}
