//! Apple-GPU blake2s kernels for the lifted Merkle tree.
//!
//! Each GPU thread hashes one leaf row (or one child pair). Columns are absorbed in
//! chunks of up to 16 (one blake message block) per dispatch, carrying the running
//! state in a GPU buffer, so any column count fits within Metal's buffer-binding
//! limit. Column-major inputs give consecutive threads consecutive addresses
//! (coalesced reads). On unified memory the column allocations are bound zero-copy
//! when page-aligned, and the output is written directly into the result vector.
//! Output is bit-identical to the scalar/SIMD builders; reference tests pin it.

use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::core::utils::uninit_vec;
use crate::core::vcs::blake2_hash::Blake2sHash;

/// Blake message block width, in u32 column values.
const CHUNK_COLS: usize = 16;
/// Minimum leaf count for GPU dispatch; below this, launch overhead dominates.
pub(crate) const MIN_METAL_LOG_SIZE: u32 = 17;

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

constant uchar SIGMA[10][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
    {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
    {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
    {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
    {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
    {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
    {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
    {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0}
};

inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }

inline void g(thread uint* v, uint a, uint b, uint c, uint d, uint x, uint y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr(v[b] ^ v[c], 7);
}

inline void compress(thread uint* h, thread const uint* m, uint t, bool last) {
    uint v[16];
    for (uint i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = IV[i]; }
    v[12] ^= t;
    if (last) { v[14] = ~v[14]; }
    for (uint r = 0; r < 10; r++) {
        constant const uchar* s = SIGMA[r];
        g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (uint i = 0; i < 8; i++) { h[i] ^= v[i] ^ v[i + 8]; }
}

constant uint M31_P = 0x7FFFFFFFu;

struct AbsorbParams {
    uint n_rows;
    // Number of valid column buffers in this chunk (1..=16); missing slots pad with 0.
    uint n_cols;
    // Running byte count BEFORE this chunk's block is absorbed.
    uint byte_count_base;
    // bit0: first chunk (initialize state); bit1: last chunk (finalize and emit).
    uint mode;
    // Reduce output words modulo M31's prime (the Blake2sM31 hasher variant).
    uint is_m31_output;
};

kernel void absorb_chunk(
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
    device uint* out [[buffer(17)]],
    constant AbsorbParams& p [[buffer(18)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n_rows) { return; }
    device const uint* cols[16] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15};

    uint h[8];
    if (p.mode & 1u) {
        for (uint k = 0; k < 8; k++) { h[k] = IV[k]; }
        h[0] ^= 0x01010020u; // digest_length 32, fanout 1, depth 1
    } else {
        for (uint k = 0; k < 8; k++) { h[k] = state[i * 8 + k]; }
    }

    uint m[16];
    for (uint j = 0; j < 16; j++) { m[j] = (j < p.n_cols) ? cols[j][i] : 0u; }
    bool last = (p.mode & 2u) != 0u;
    uint t = p.byte_count_base + p.n_cols * 4u;
    compress(h, m, t, last);

    if (!last) {
        for (uint k = 0; k < 8; k++) { state[i * 8 + k] = h[k]; }
        return;
    }
    for (uint k = 0; k < 8; k++) {
        uint v = h[k];
        if (p.is_m31_output != 0u) {
            uint r = (v & M31_P) + (v >> 31);
            if (r >= M31_P) { r -= M31_P; }
            v = r;
        }
        out[i * 8 + k] = v;
    }
}

kernel void hash_children(
    device const uint* prev [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant uint& n_out [[buffer(2)]],
    constant uint& is_m31_output [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n_out) { return; }
    uint h[8];
    for (uint k = 0; k < 8; k++) { h[k] = IV[k]; }
    h[0] ^= 0x01010020u;
    uint m[16];
    for (uint j = 0; j < 16; j++) { m[j] = prev[i * 16 + j]; }
    compress(h, m, 64u, true);
    for (uint k = 0; k < 8; k++) {
        uint v = h[k];
        if (is_m31_output != 0u) {
            uint r = (v & M31_P) + (v >> 31);
            if (r >= M31_P) { r -= M31_P; }
            v = r;
        }
        out[i * 8 + k] = v;
    }
}
"#;

struct MetalContext {
    device: Device,
    queue: CommandQueue,
    absorb_pipeline: ComputePipelineState,
    children_pipeline: ComputePipelineState,
    /// Reused running-state buffer (n_rows x 8 u32), grown on demand.
    state_buffer: Option<Buffer>,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for MetalContext {}

/// Forces device/pipeline initialization; see [`super::warmup`].
pub(crate) fn warmup() {
    let _ = context();
}

fn context() -> Option<&'static Mutex<MetalContext>> {
    static CONTEXT: OnceLock<Option<Mutex<MetalContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let device = Device::system_default()?;
            if !device.has_unified_memory() {
                return None;
            }
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let absorb = library.get_function("absorb_chunk", None).ok()?;
            let children = library.get_function("hash_children", None).ok()?;
            let absorb_pipeline = device
                .new_compute_pipeline_state_with_function(&absorb)
                .ok()?;
            let children_pipeline = device
                .new_compute_pipeline_state_with_function(&children)
                .ok()?;
            let queue = device.new_command_queue();
            Some(Mutex::new(MetalContext {
                device,
                queue,
                absorb_pipeline,
                children_pipeline,
                state_buffer: None,
            }))
        })
        .as_ref()
}

/// Wraps an existing allocation as a zero-copy shared buffer when it satisfies the
/// page alignment Metal requires; otherwise copies it into a fresh shared buffer.
fn input_buffer(device: &Device, data: &[u32]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let page = 16384;
    if (data.as_ptr() as usize).is_multiple_of(page) && bytes.is_multiple_of(page) {
        // The caller keeps `data` alive (and unmodified) until the command buffer
        // completes; completion is awaited before returning to the caller.
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

/// Faults a fresh allocation's pages in on the CPU (parallel) so GPU kernels don't
/// stall on first-touch page faults; contents are fully overwritten by the kernels.
fn prefault<T: Send>(data: &mut [T]) {
    let page_elems = 16384 / std::mem::size_of::<T>();
    let fill = |chunk: &mut [T]| {
        for slot in chunk.iter_mut().step_by(page_elems) {
            // Safety: writing zero bytes into allocated memory of any plain type.
            unsafe { std::ptr::write_bytes(slot, 0, 1) };
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(page_elems * 256).for_each(fill);
    }
    #[cfg(not(feature = "parallel"))]
    fill(data);
}

/// Binds the result vector as a zero-copy shared buffer when page-aligned, else
/// allocates a shared scratch buffer the caller copies out of after completion.
/// Returns (buffer, is_zero_copy).
fn output_buffer(device: &Device, res: &mut [Blake2sHash]) -> (Buffer, bool) {
    let bytes = res.len() * 32;
    let page = 16384;
    if (res.as_ptr() as usize).is_multiple_of(page) && bytes.is_multiple_of(page) {
        // The vector outlives the awaited command buffer.
        let buffer = device.new_buffer_with_bytes_no_copy(
            res.as_ptr() as *const std::ffi::c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        (buffer, true)
    } else {
        (
            device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            false,
        )
    }
}

#[repr(C)]
struct AbsorbParams {
    n_rows: u32,
    n_cols: u32,
    byte_count_base: u32,
    mode: u32,
    is_m31_output: u32,
}

/// GPU leaf builder for same-size columns. Returns `None` when no usable device is
/// present; callers must fall back to the CPU/SIMD builders. The output equals
/// [`crate::prover::backend::simd::blake2s_lifted::build_leaves_from_flat_columns`]
/// bit for bit.
pub(crate) fn build_leaves_metal(
    columns: &[&[u32]],
    is_m31_output: bool,
) -> Option<Vec<Blake2sHash>> {
    let n_rows = columns[0].len();
    if columns.iter().any(|c| c.len() != n_rows) {
        return None;
    }
    let ctx = context()?;
    let mut ctx = ctx.lock().unwrap();

    // Safety: every entry is written by the final GPU chunk before being read.
    let mut res: Vec<Blake2sHash> = unsafe { uninit_vec(n_rows) };
    prefault(&mut res);

    let state_bytes = (n_rows * 32) as u64;
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
        .map(|c| input_buffer(&ctx.device, c))
        .collect();
    let (out_buffer, out_zero_copy) = output_buffer(&ctx.device, &mut res);

    let command_buffer = ctx.queue.new_command_buffer();
    let n_chunks = columns.len().div_ceil(CHUNK_COLS);
    let mut byte_count = 0u32;
    for (chunk_idx, chunk) in in_buffers.chunks(CHUNK_COLS).enumerate() {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.absorb_pipeline);
        for slot in 0..CHUNK_COLS {
            // Unused slots must still be bound; alias the first column (never read).
            let buffer = chunk.get(slot).unwrap_or(&chunk[0]);
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_buffer(16, ctx.state_buffer.as_ref().map(|b| b as _), 0);
        encoder.set_buffer(17, Some(&out_buffer), 0);
        let params = AbsorbParams {
            n_rows: n_rows as u32,
            n_cols: chunk.len() as u32,
            byte_count_base: byte_count,
            mode: u32::from(chunk_idx == 0) | (u32::from(chunk_idx == n_chunks - 1) << 1),
            is_m31_output: u32::from(is_m31_output),
        };
        encoder.set_bytes(
            18,
            std::mem::size_of::<AbsorbParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        byte_count += chunk.len() as u32 * 4;
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if !out_zero_copy {
        // Safety: the kernel wrote all `n_rows` hashes into the shared buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buffer.contents() as *const Blake2sHash,
                res.as_mut_ptr(),
                n_rows,
            );
        }
    }

    Some(res)
}

/// GPU Merkle layer step: hashes consecutive child pairs of `prev_layer`. Returns
/// `None` when no usable device is present.
pub(crate) fn build_next_layer_metal(
    prev_layer: &[Blake2sHash],
    is_m31_output: bool,
) -> Option<Vec<Blake2sHash>> {
    let n_out = prev_layer.len() / 2;
    if n_out < (1 << MIN_METAL_LOG_SIZE) {
        return None;
    }
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    // Safety: every entry is written by the kernel before being read.
    let mut res: Vec<Blake2sHash> = unsafe { uninit_vec(n_out) };
    prefault(&mut res);
    let prev_words = unsafe {
        std::slice::from_raw_parts(prev_layer.as_ptr() as *const u32, prev_layer.len() * 8)
    };
    let in_buffer = input_buffer(&ctx.device, prev_words);
    let (out_buffer, out_zero_copy) = output_buffer(&ctx.device, &mut res);

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&ctx.children_pipeline);
    encoder.set_buffer(0, Some(&in_buffer), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    let n = n_out as u32;
    encoder.set_bytes(2, 4, &n as *const _ as *const std::ffi::c_void);
    let m31 = u32::from(is_m31_output);
    encoder.set_bytes(3, 4, &m31 as *const _ as *const std::ffi::c_void);
    encoder.dispatch_threads(MTLSize::new(n_out as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if !out_zero_copy {
        // Safety: the kernel wrote all `n_out` hashes into the shared buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buffer.contents() as *const Blake2sHash,
                res.as_mut_ptr(),
                n_out,
            );
        }
    }

    Some(res)
}

/// Builds all `n_layers` Merkle layers above `leaves` in one GPU submission (one
/// synchronization for the whole chain); layers below the dispatch threshold are
/// finished by the caller's fallback. Returns `None` when no usable device exists,
/// passing `leaves` back untouched via the `Err`-like option contract (callers keep
/// ownership by cloning nothing: `leaves` is returned as the first layer on success).
pub(crate) fn build_layers_metal(
    leaves: Vec<Blake2sHash>,
    n_layers: u32,
    is_m31_output: bool,
) -> Result<Vec<Vec<Blake2sHash>>, Vec<Blake2sHash>> {
    if (leaves.len() / 2) < (1 << MIN_METAL_LOG_SIZE) {
        return Err(leaves);
    }
    let Some(ctx) = context() else {
        return Err(leaves);
    };
    let ctx = ctx.lock().unwrap();

    // GPU levels: every level whose output is still >= the dispatch threshold.
    let mut sizes = vec![];
    let mut n = leaves.len() / 2;
    while n >= (1 << MIN_METAL_LOG_SIZE) && (sizes.len() as u32) < n_layers {
        sizes.push(n);
        n /= 2;
    }
    // Safety: every entry of every level is written by its kernel before being read.
    let mut levels: Vec<Vec<Blake2sHash>> =
        sizes.iter().map(|&n| unsafe { uninit_vec(n) }).collect();
    levels.iter_mut().for_each(|level| prefault(level));

    let command_buffer = ctx.queue.new_command_buffer();
    let mut prev_buffer = {
        let words =
            unsafe { std::slice::from_raw_parts(leaves.as_ptr() as *const u32, leaves.len() * 8) };
        input_buffer(&ctx.device, words)
    };
    let mut copy_outs = vec![];
    for level in levels.iter_mut() {
        let (out_buffer, zero_copy) = output_buffer(&ctx.device, level);
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.children_pipeline);
        encoder.set_buffer(0, Some(&prev_buffer), 0);
        encoder.set_buffer(1, Some(&out_buffer), 0);
        let n = level.len() as u32;
        encoder.set_bytes(2, 4, &n as *const _ as *const std::ffi::c_void);
        let m31 = u32::from(is_m31_output);
        encoder.set_bytes(3, 4, &m31 as *const _ as *const std::ffi::c_void);
        encoder.dispatch_threads(
            MTLSize::new(level.len() as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        if !zero_copy {
            copy_outs.push((out_buffer.clone(), level.len()));
        } else {
            copy_outs.push((out_buffer.clone(), 0));
        }
        prev_buffer = out_buffer;
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();
    for ((buffer, copy_len), level) in copy_outs.into_iter().zip(levels.iter_mut()) {
        if copy_len > 0 {
            // Safety: the kernel wrote all entries into the shared buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const Blake2sHash,
                    level.as_mut_ptr(),
                    copy_len,
                );
            }
        }
    }

    let mut layers = vec![leaves];
    layers.extend(levels);
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::backend::simd::blake2s_lifted::{
        build_leaves_from_flat_columns, build_next_layer_simd,
    };

    fn test_columns(n_cols: usize, log_rows: u32) -> Vec<Vec<u32>> {
        (0..n_cols)
            .map(|c| {
                (0..1u32 << log_rows)
                    .map(|i| (i.wrapping_mul(2654435761).wrapping_add(c as u32 * 97)) >> 1)
                    .collect()
            })
            .collect()
    }

    /// The GPU leaf builder must be bit-identical to the SIMD reference, for both
    /// hasher variants and for column counts around the message-block boundary.
    #[test]
    fn metal_leaves_match_simd_reference() {
        const LOG_ROWS: u32 = MIN_METAL_LOG_SIZE;
        for n_cols in [4usize, 16, 17, 104] {
            let columns = test_columns(n_cols, LOG_ROWS);
            let refs: Vec<&[u32]> = columns.iter().map(|c| c.as_slice()).collect();
            for is_m31 in [false, true] {
                let Some(gpu) = build_leaves_metal(&refs, is_m31) else {
                    // No usable GPU on this machine; the dispatch falls back.
                    return;
                };
                let reference = if is_m31 {
                    build_leaves_from_flat_columns::<true>(&refs, LOG_ROWS)
                } else {
                    build_leaves_from_flat_columns::<false>(&refs, LOG_ROWS)
                };
                assert_eq!(gpu, reference, "n_cols {n_cols} is_m31 {is_m31}");
            }
        }
    }

    /// The GPU layer step must be bit-identical to the SIMD reference.
    #[test]
    fn metal_next_layer_matches_simd_reference() {
        let columns = test_columns(4, MIN_METAL_LOG_SIZE + 1);
        let refs: Vec<&[u32]> = columns.iter().map(|c| c.as_slice()).collect();
        let leaves = build_leaves_from_flat_columns::<true>(&refs, MIN_METAL_LOG_SIZE + 1);
        let Some(gpu) = build_next_layer_metal(&leaves, true) else {
            return;
        };
        assert_eq!(gpu, build_next_layer_simd::<true>(&leaves));
    }
}
