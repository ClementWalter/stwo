//! Apple-GPU blake2s kernels for the lifted Merkle tree.
//!
//! Each GPU thread hashes one leaf row (or one child pair). Columns are absorbed in
//! chunks of up to 16 (one blake message block) per dispatch, carrying the running
//! state in the output buffer, so any column count fits within Metal's buffer-binding
//! limit without a second hash-sized allocation. Column-major inputs give consecutive
//! threads consecutive addresses (coalesced reads). On unified memory the column
//! allocations are bound zero-copy when page-aligned, and the output is written
//! directly into the result vector. Output is bit-identical to the scalar/SIMD
//! builders; reference tests pin it.

use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

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
    device uint* state_out [[buffer(16)]],
    constant AbsorbParams& p [[buffer(17)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n_rows) { return; }
    device const uint* cols[16] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15};

    uint h[8];
    if (p.mode & 1u) {
        for (uint k = 0; k < 8; k++) { h[k] = IV[k]; }
        h[0] ^= 0x01010020u; // digest_length 32, fanout 1, depth 1
    } else {
        for (uint k = 0; k < 8; k++) { h[k] = state_out[i * 8 + k]; }
    }

    uint m[16];
    for (uint j = 0; j < 16; j++) { m[j] = (j < p.n_cols) ? cols[j][i] : 0u; }
    bool last = (p.mode & 2u) != 0u;
    uint t = p.byte_count_base + p.n_cols * 4u;
    compress(h, m, t, last);

    if (!last) {
        for (uint k = 0; k < 8; k++) { state_out[i * 8 + k] = h[k]; }
        return;
    }
    // This thread loaded its entire previous state above. Its final output may
    // therefore overwrite the same disjoint 8-word range without a second buffer.
    for (uint k = 0; k < 8; k++) {
        uint v = h[k];
        if (p.is_m31_output != 0u) {
            uint r = (v & M31_P) + (v >> 31);
            if (r >= M31_P) { r -= M31_P; }
            v = r;
        }
        state_out[i * 8 + k] = v;
    }
}

inline uint lifted_index(uint index, uint log_ratio) {
    return ((index >> (log_ratio + 1u)) << 1u) | (index & 1u);
}

struct CompactAbsorbParams {
    uint n_rows;
    uint n_cols;
    uint first_column;
    uint mode;
    uint is_m31_output;
    uint source_state_log;
    uint destination_log;
    uint column_logs[16];
};

// Absorbs one canonical 16-column Blake block at the smallest useful row count.
// When the block's destination log grows, the prior running state is lifted with
// the same two-coset index map used by the lifted Merkle verifier.
kernel void absorb_chunk_compact(
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
    device const uint* source_state [[buffer(16)]],
    device uint* destination_state [[buffer(17)]],
    constant CompactAbsorbParams& p [[buffer(18)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n_rows) { return; }
    device const uint* cols[16] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15};

    uint h[8];
    if (p.mode & 1u) {
        for (uint k = 0; k < 8; k++) { h[k] = IV[k]; }
        h[0] ^= 0x01010020u;
    } else {
        uint source_i = lifted_index(i, p.destination_log - p.source_state_log);
        for (uint k = 0; k < 8; k++) { h[k] = source_state[source_i * 8u + k]; }
    }

    uint m[16];
    for (uint j = 0; j < 16; j++) {
        uint source_i = lifted_index(i, p.destination_log - p.column_logs[j]);
        m[j] = (j < p.n_cols) ? cols[j][source_i] : 0u;
    }
    bool last = (p.mode & 2u) != 0u;
    compress(h, m, (p.first_column + p.n_cols) * 4u, last);

    for (uint k = 0; k < 8; k++) {
        uint v = h[k];
        if (last && p.is_m31_output != 0u) {
            uint r = (v & M31_P) + (v >> 31);
            if (r >= M31_P) { r -= M31_P; }
            v = r;
        }
        destination_state[i * 8u + k] = v;
    }
}

// One packed leaf = 16 M31 values = one blake block: leaf i absorbs
// coords[c % 4][4i + c / 4] for c in 0..16 (the packed-leaf layout).
kernel void packed_leaves(
    device const uint* c0 [[buffer(0)]],
    device const uint* c1 [[buffer(1)]],
    device const uint* c2 [[buffer(2)]],
    device const uint* c3 [[buffer(3)]],
    device uint* out [[buffer(4)]],
    constant uint& n_leaves [[buffer(5)]],
    constant uint& is_m31_output [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n_leaves) { return; }
    device const uint* coords[4] = {c0, c1, c2, c3};
    uint h[8];
    for (uint k = 0; k < 8; k++) { h[k] = IV[k]; }
    h[0] ^= 0x01010020u;
    uint m[16];
    for (uint c = 0; c < 16u; c++) { m[c] = coords[c & 3u][(i << 2u) + (c >> 2u)]; }
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
    compact_absorb_pipeline: ComputePipelineState,
    children_pipeline: ComputePipelineState,
    packed_leaves_pipeline: ComputePipelineState,
}

// Metal objects are reference-counted Objective-C handles; all uses are serialized
// behind the context mutex.
unsafe impl Send for MetalContext {}

/// Forces device/pipeline initialization; see [`super::warmup`].
pub(crate) fn warmup() {
    let _ = context();
}

/// Returns whether the static Blake2s pipelines compiled successfully.
pub(crate) fn is_ready() -> bool {
    let Some(context) = context() else {
        return false;
    };
    context.lock().is_ok()
}

fn context() -> Option<&'static Mutex<MetalContext>> {
    static CONTEXT: OnceLock<Option<Mutex<MetalContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let shared = super::context::gpu()?;
            let device = shared.device.clone();
            let library = device
                .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
                .ok()?;
            let absorb = library.get_function("absorb_chunk", None).ok()?;
            let compact_absorb = library.get_function("absorb_chunk_compact", None).ok()?;
            let children = library.get_function("hash_children", None).ok()?;
            let packed = library.get_function("packed_leaves", None).ok()?;
            let absorb_pipeline = device
                .new_compute_pipeline_state_with_function(&absorb)
                .ok()?;
            let compact_absorb_pipeline = device
                .new_compute_pipeline_state_with_function(&compact_absorb)
                .ok()?;
            let children_pipeline = device
                .new_compute_pipeline_state_with_function(&children)
                .ok()?;
            let packed_leaves_pipeline = device
                .new_compute_pipeline_state_with_function(&packed)
                .ok()?;
            let queue = shared.queue.clone();
            Some(Mutex::new(MetalContext {
                device,
                queue,
                absorb_pipeline,
                compact_absorb_pipeline,
                children_pipeline,
                packed_leaves_pipeline,
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

/// Binds the result vector as a zero-copy shared buffer when page-aligned, else
/// allocates a shared scratch buffer the caller copies out of after completion.
/// Returns (buffer, is_zero_copy).
fn output_buffer(device: &Device, res: &mut [Blake2sHash]) -> (Buffer, bool) {
    let bytes = res.len() * 32;
    let page = 16384;
    if (res.as_mut_ptr() as usize).is_multiple_of(page) && bytes.is_multiple_of(page) {
        // The exclusive borrow is not used again until the awaited command completes.
        let buffer = device.new_buffer_with_bytes_no_copy(
            res.as_mut_ptr() as *const std::ffi::c_void,
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

#[repr(C)]
struct CompactAbsorbParams {
    n_rows: u32,
    n_cols: u32,
    first_column: u32,
    mode: u32,
    is_m31_output: u32,
    source_state_log: u32,
    destination_log: u32,
    column_logs: [u32; CHUNK_COLS],
}

/// GPU leaf builder for same-size columns. Returns `None` when no usable device is
/// present or execution fails; callers must fall back to the CPU/SIMD builders. The output equals
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
    let ctx = ctx.lock().unwrap();

    let mut res = vec![Blake2sHash::default(); n_rows];

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
        encoder.set_buffer(16, Some(&out_buffer), 0);
        let params = AbsorbParams {
            n_rows: n_rows as u32,
            n_cols: chunk.len() as u32,
            byte_count_base: byte_count,
            mode: u32::from(chunk_idx == 0) | (u32::from(chunk_idx == n_chunks - 1) << 1),
            is_m31_output: u32::from(is_m31_output),
        };
        encoder.set_bytes(
            17,
            std::mem::size_of::<AbsorbParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        byte_count += chunk.len() as u32 * 4;
    }
    command_buffer.commit();
    super::context::wait_for_completion(command_buffer).ok()?;
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
/// `None` when no usable device is present or execution fails.
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

    let mut res = vec![Blake2sHash::default(); n_out];
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
    super::context::wait_for_completion(command_buffer).ok()?;
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
/// A command failure also returns the untouched input leaves through `Err`.
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
    let mut levels: Vec<Vec<Blake2sHash>> = sizes
        .iter()
        .map(|&n| vec![Blake2sHash::default(); n])
        .collect();

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
    if super::context::wait_for_completion(command_buffer).is_err() {
        return Err(leaves);
    }
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

/// Tree outputs of [`encode_tree`], finalized after the caller's submission completes.
pub(crate) struct PendingTree {
    pub leaves: Vec<Blake2sHash>,
    levels: Vec<Vec<Blake2sHash>>,
    copy_outs: Vec<(Buffer, usize)>,
    // Compact mixed-log absorption can require exact-sized rolling-state buffers.
    // Retain them until the caller has waited for the enclosing command buffer.
    _intermediate_states: Vec<Buffer>,
}

impl PendingTree {
    /// Resolves any non-zero-copy level buffers; call only after the submission that
    /// ran the encoded kernels has completed successfully.
    pub(crate) fn finish(mut self) -> Vec<Vec<Blake2sHash>> {
        for ((buffer, copy_len), level) in self.copy_outs.iter().zip(self.levels.iter_mut()) {
            if *copy_len > 0 {
                // Safety: the kernel wrote all entries into the shared buffer.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.contents() as *const Blake2sHash,
                        level.as_mut_ptr(),
                        *copy_len,
                    );
                }
            }
        }
        let mut layers = vec![self.leaves];
        layers.extend(self.levels);
        layers
    }
}

/// Encodes the whole Merkle tree (uniform-size leaves + every above-threshold layer)
/// into the caller's command buffer, reading the already-bound LDE column buffers.
/// Sub-threshold tail layers must be finished by the caller after the wait. Returns
/// `None` when no usable device exists.
pub(crate) fn encode_tree(
    command_buffer: &metal::CommandBufferRef,
    column_buffers: &[Buffer],
    n_rows: usize,
    is_m31_output: bool,
) -> Option<PendingTree> {
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    // Leaves: chunked absorption, exactly as build_leaves_metal.
    let mut leaves = vec![Blake2sHash::default(); n_rows];
    let (leaves_buffer, leaves_zero_copy) = output_buffer(&ctx.device, &mut leaves);
    if !leaves_zero_copy {
        // The chain relies on zero-copy outputs throughout; bail to the separate path.
        return None;
    }
    let n_chunks = column_buffers.len().div_ceil(CHUNK_COLS);
    let mut byte_count = 0u32;
    for (chunk_idx, chunk) in column_buffers.chunks(CHUNK_COLS).enumerate() {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.absorb_pipeline);
        for slot in 0..CHUNK_COLS {
            // Unused slots must still be bound; alias the first column (never read).
            let buffer = chunk.get(slot).unwrap_or(&chunk[0]);
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_buffer(16, Some(&leaves_buffer), 0);
        let params = AbsorbParams {
            n_rows: n_rows as u32,
            n_cols: chunk.len() as u32,
            byte_count_base: byte_count,
            mode: u32::from(chunk_idx == 0) | (u32::from(chunk_idx == n_chunks - 1) << 1),
            is_m31_output: u32::from(is_m31_output),
        };
        encoder.set_bytes(
            17,
            std::mem::size_of::<AbsorbParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        byte_count += chunk.len() as u32 * 4;
    }

    // Layer chain, exactly as build_layers_metal.
    let mut sizes = vec![];
    let mut n = n_rows / 2;
    while n >= (1 << MIN_METAL_LOG_SIZE) {
        sizes.push(n);
        n /= 2;
    }
    let mut levels: Vec<Vec<Blake2sHash>> = sizes
        .iter()
        .map(|&n| vec![Blake2sHash::default(); n])
        .collect();
    let mut prev_buffer = leaves_buffer;
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
        copy_outs.push((out_buffer.clone(), if zero_copy { 0 } else { level.len() }));
        prev_buffer = out_buffer;
    }

    Some(PendingTree {
        leaves,
        levels,
        copy_outs,
        _intermediate_states: vec![],
    })
}

/// Encodes a lifted Merkle tree for columns with distinct log sizes. `column_buffers`
/// must be in the protocol's stable `(log_size, original_index)` order. Running Blake
/// state is materialized only at the largest log needed by each canonical 16-column
/// block, then lifted when a later block grows; this avoids hashing every small column
/// independently at the final lifting size.
pub(crate) fn encode_tree_compact(
    command_buffer: &metal::CommandBufferRef,
    column_buffers: &[Buffer],
    column_logs: &[u32],
    lifting_log: u32,
    is_m31_output: bool,
) -> Option<PendingTree> {
    if column_buffers.is_empty()
        || column_buffers.len() != column_logs.len()
        || column_logs.iter().any(|&log| log == 0 || log > lifting_log)
        || column_logs.windows(2).any(|logs| logs[0] > logs[1])
        || lifting_log >= u32::BITS
        || column_buffers.len() > (u32::MAX / 4) as usize
    {
        return None;
    }
    let n_rows = 1usize.checked_shl(lifting_log)?;
    for (buffer, &log) in column_buffers.iter().zip(column_logs) {
        let expected_bytes = 1u64.checked_shl(log)?.checked_mul(4)?;
        if buffer.length() != expected_bytes {
            return None;
        }
    }
    // Resolve every fallible size/counter operation before touching the caller's
    // command buffer. Returning None must never leave a partial commitment epoch.
    let n_chunks = column_buffers.len().div_ceil(CHUNK_COLS);
    let mut stages = Vec::with_capacity(n_chunks);
    for (chunk_idx, chunk_logs) in column_logs.chunks(CHUNK_COLS).enumerate() {
        let last = chunk_idx + 1 == n_chunks;
        let destination_log = if last {
            lifting_log
        } else {
            *chunk_logs.iter().max()?
        };
        let stage_rows = 1u32.checked_shl(destination_log)?;
        let state_bytes = u64::from(stage_rows).checked_mul(32)?;
        let first_column = u32::try_from(chunk_idx.checked_mul(CHUNK_COLS)?).ok()?;
        stages.push((destination_log, stage_rows, state_bytes, first_column));
    }
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    let mut leaves = vec![Blake2sHash::default(); n_rows];
    let (leaves_buffer, leaves_zero_copy) = output_buffer(&ctx.device, &mut leaves);
    if !leaves_zero_copy {
        return None;
    }

    let mut source_state: Option<(Buffer, u32)> = None;
    let mut intermediate_states = vec![];
    for (chunk_idx, ((chunk, chunk_logs), &stage)) in column_buffers
        .chunks(CHUNK_COLS)
        .zip(column_logs.chunks(CHUNK_COLS))
        .zip(&stages)
        .enumerate()
    {
        let (destination_log, stage_rows, state_bytes, first_column) = stage;
        let first = chunk_idx == 0;
        let last = chunk_idx + 1 == n_chunks;
        let destination_state = if last {
            leaves_buffer.clone()
        } else if let Some((source, source_log)) = &source_state {
            if *source_log == destination_log {
                source.clone()
            } else {
                let buffer = ctx
                    .device
                    .new_buffer(state_bytes, MTLResourceOptions::StorageModePrivate);
                intermediate_states.push(buffer.clone());
                buffer
            }
        } else {
            let buffer = ctx
                .device
                .new_buffer(state_bytes, MTLResourceOptions::StorageModePrivate);
            intermediate_states.push(buffer.clone());
            buffer
        };
        let source_buffer = source_state
            .as_ref()
            .map_or(&destination_state, |(buffer, _)| buffer);
        let source_log = source_state
            .as_ref()
            .map_or(destination_log, |(_, log)| *log);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.compact_absorb_pipeline);
        for slot in 0..CHUNK_COLS {
            // Metal requires every declared resource to be bound. Missing message
            // words are masked by n_cols, so aliasing slot zero is safe.
            let buffer = chunk.get(slot).unwrap_or(&chunk[0]);
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_buffer(16, Some(source_buffer), 0);
        encoder.set_buffer(17, Some(&destination_state), 0);
        let mut logs = [destination_log; CHUNK_COLS];
        logs[..chunk_logs.len()].copy_from_slice(chunk_logs);
        let params = CompactAbsorbParams {
            n_rows: stage_rows,
            n_cols: chunk.len() as u32,
            first_column,
            mode: u32::from(first) | (u32::from(last) << 1),
            is_m31_output: u32::from(is_m31_output),
            source_state_log: source_log,
            destination_log,
            column_logs: logs,
        };
        encoder.set_bytes(
            18,
            std::mem::size_of::<CompactAbsorbParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(
            MTLSize::new(params.n_rows as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        source_state = Some((destination_state, destination_log));
    }

    // Chain every tree layer that is still large enough to amortize a GPU launch.
    let mut sizes = vec![];
    let mut n = n_rows / 2;
    while n >= (1 << MIN_METAL_LOG_SIZE) {
        sizes.push(n);
        n /= 2;
    }
    let mut levels: Vec<Vec<Blake2sHash>> = sizes
        .iter()
        .map(|&len| vec![Blake2sHash::default(); len])
        .collect();
    let mut prev_buffer = leaves_buffer;
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
        copy_outs.push((out_buffer.clone(), if zero_copy { 0 } else { level.len() }));
        prev_buffer = out_buffer;
    }

    Some(PendingTree {
        leaves,
        levels,
        copy_outs,
        _intermediate_states: intermediate_states,
    })
}

/// Builds the whole packed-leaf Merkle tree (FRI layer shape: four secure-coordinate
/// columns, four rows per leaf) in one submission: the leaf kernel reads the
/// coordinate columns directly in packed order — no packing pass — and the layer
/// chain follows. Returns layers leaves-first; `None` without a usable device.
pub(crate) fn build_packed_tree_metal(
    coords: [&[crate::core::fields::m31::BaseField]; 4],
    is_m31_output: bool,
) -> Option<Vec<Vec<Blake2sHash>>> {
    let n_leaves = coords[0].len() / 4;
    if n_leaves < (1 << MIN_METAL_LOG_SIZE) {
        return None;
    }
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    let coord_buffers: Vec<Buffer> = coords
        .iter()
        .map(|c| {
            // BaseField is a transparent u32 wrapper.
            let words = unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u32, c.len()) };
            input_buffer(&ctx.device, words)
        })
        .collect();
    let command_buffer = ctx.queue.new_command_buffer();
    let pending = encode_packed_tree_inner(
        &ctx,
        command_buffer,
        &coord_buffers,
        n_leaves,
        is_m31_output,
    )?;
    command_buffer.commit();
    super::context::wait_for_completion(command_buffer).ok()?;
    Some(pending.finish())
}

/// Encodes the packed-leaf kernel and the layer chain into the caller's command
/// buffer; see [`build_packed_tree_metal`].
pub(crate) fn encode_packed_tree(
    command_buffer: &metal::CommandBufferRef,
    coord_buffers: &[Buffer],
    n_leaves: usize,
    is_m31_output: bool,
) -> Option<PendingTree> {
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();
    encode_packed_tree_inner(&ctx, command_buffer, coord_buffers, n_leaves, is_m31_output)
}

fn encode_packed_tree_inner(
    ctx: &MetalContext,
    command_buffer: &metal::CommandBufferRef,
    coord_buffers: &[Buffer],
    n_leaves: usize,
    is_m31_output: bool,
) -> Option<PendingTree> {
    let mut leaves = vec![Blake2sHash::default(); n_leaves];
    let (leaves_buffer, leaves_zero_copy) = output_buffer(&ctx.device, &mut leaves);
    if !leaves_zero_copy {
        return None;
    }
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.packed_leaves_pipeline);
        for (k, buffer) in coord_buffers.iter().enumerate() {
            encoder.set_buffer(k as u64, Some(buffer), 0);
        }
        encoder.set_buffer(4, Some(&leaves_buffer), 0);
        let n = n_leaves as u32;
        encoder.set_bytes(5, 4, &n as *const _ as *const std::ffi::c_void);
        let m31 = u32::from(is_m31_output);
        encoder.set_bytes(6, 4, &m31 as *const _ as *const std::ffi::c_void);
        encoder.dispatch_threads(MTLSize::new(n_leaves as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }

    // Layer chain over the leaves, as in build_layers_metal.
    let mut sizes = vec![];
    let mut n = n_leaves / 2;
    while n >= (1 << MIN_METAL_LOG_SIZE) {
        sizes.push(n);
        n /= 2;
    }
    let mut levels: Vec<Vec<Blake2sHash>> = sizes
        .iter()
        .map(|&n| vec![Blake2sHash::default(); n])
        .collect();
    let mut prev_buffer = leaves_buffer;
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
        copy_outs.push((out_buffer.clone(), if zero_copy { 0 } else { level.len() }));
        prev_buffer = out_buffer;
    }
    Some(PendingTree {
        leaves,
        levels,
        copy_outs,
        _intermediate_states: vec![],
    })
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

    fn metal_leaves_match_simd_reference(n_cols: usize) -> Option<()> {
        const LOG_ROWS: u32 = MIN_METAL_LOG_SIZE;
        let columns = test_columns(n_cols, LOG_ROWS);
        let refs: Vec<&[u32]> = columns.iter().map(|c| c.as_slice()).collect();
        for is_m31 in [false, true] {
            let gpu = build_leaves_metal(&refs, is_m31)?;
            let reference = if is_m31 {
                build_leaves_from_flat_columns::<true>(&refs, LOG_ROWS)
            } else {
                build_leaves_from_flat_columns::<false>(&refs, LOG_ROWS)
            };
            assert_eq!(gpu, reference, "n_cols {n_cols} is_m31 {is_m31}");
        }
        Some(())
    }

    /// A single dispatch must be bit-identical to the CPU SIMD reference for both
    /// hasher variants, including a partial and a full Blake message block.
    #[test]
    fn metal_one_chunk_leaves_match_simd_reference() {
        for n_cols in [4, CHUNK_COLS] {
            if metal_leaves_match_simd_reference(n_cols).is_none() {
                // No usable GPU on this machine; the dispatch falls back.
                return;
            }
        }
    }

    /// Reusing the output allocation as rolling state across two or many dispatches
    /// must remain bit-identical to the CPU SIMD reference for both hasher variants.
    #[test]
    fn metal_multiple_chunk_leaves_match_simd_reference() {
        for n_cols in [CHUNK_COLS + 1, 104] {
            if metal_leaves_match_simd_reference(n_cols).is_none() {
                // No usable GPU on this machine; the dispatch falls back.
                return;
            }
        }
    }

    fn assert_compact_tree_matches_reference(logs: &[u32], lifting_log: u32) {
        let columns: Vec<Vec<u32>> = logs
            .iter()
            .enumerate()
            .map(|(column, &log)| {
                (0..1u32 << log)
                    .map(|row| {
                        row.wrapping_mul(2654435761)
                            .wrapping_add(column as u32 * 97)
                            >> 1
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[u32]> = columns.iter().map(Vec::as_slice).collect();

        for is_m31 in [false, true] {
            let (buffers, command_buffer) = {
                let ctx = context().expect("Metal context must initialize for strict test");
                let ctx = ctx.lock().unwrap();
                let buffers = refs
                    .iter()
                    .map(|column| input_buffer(&ctx.device, column))
                    .collect::<Vec<_>>();
                (buffers, ctx.queue.new_command_buffer().to_owned())
            };
            let pending = encode_tree_compact(&command_buffer, &buffers, logs, lifting_log, is_m31)
                .expect("strict compact Metal encoding must succeed");
            command_buffer.commit();
            super::super::context::wait_for_completion(&command_buffer)
                .expect("strict compact Metal dispatch must succeed");
            let mut gpu = pending.finish();
            while (gpu.len() as u32) < lifting_log + 1 {
                let next = if is_m31 {
                    build_next_layer_simd::<true>(gpu.last().unwrap())
                } else {
                    build_next_layer_simd::<false>(gpu.last().unwrap())
                };
                gpu.push(next);
            }

            let leaves = if is_m31 {
                build_leaves_from_flat_columns::<true>(&refs, lifting_log)
            } else {
                build_leaves_from_flat_columns::<false>(&refs, lifting_log)
            };
            let mut reference = vec![leaves];
            while (reference.len() as u32) < lifting_log + 1 {
                let next = if is_m31 {
                    build_next_layer_simd::<true>(reference.last().unwrap())
                } else {
                    build_next_layer_simd::<false>(reference.last().unwrap())
                };
                reference.push(next);
            }
            assert_eq!(gpu, reference, "mixed tree differs, is_m31={is_m31}");
        }
    }

    /// Compact staged absorption must match the established SIMD lifted tree for
    /// value-distinct equal-log columns, transitions inside Blake blocks, more than
    /// two chunks, a partial final block, a final lift, and both digest representations.
    #[test]
    fn metal_compact_mixed_tree_matches_simd_reference() {
        // The first block exercises lifted-index ratios 4, 3, 1, and 0; the
        // following blocks add ratio 2 and a distinct final state lift.
        let logs = [
            vec![10; 3],
            vec![11; 3],
            vec![13; 3],
            vec![14; 11],
            vec![16; 13],
        ]
        .concat();
        assert_compact_tree_matches_reference(&logs, 18);
    }

    /// Finalization at an exact Blake-block boundary must preserve the same transcript.
    #[test]
    fn metal_compact_full_final_block_matches_simd_reference() {
        let logs = [vec![12; 16], vec![16; 16]].concat();
        assert_compact_tree_matches_reference(&logs, 18);
    }

    /// Invalid layouts are rejected before any encoder can be appended to the caller's
    /// command buffer.
    #[test]
    fn metal_compact_rejects_invalid_layouts() {
        let ctx = context().expect("Metal context must initialize for strict test");
        let ctx = ctx.lock().unwrap();
        let columns = test_columns(2, 12);
        let buffers = columns
            .iter()
            .map(|column| input_buffer(&ctx.device, column))
            .collect::<Vec<_>>();
        let command_buffer = ctx.queue.new_command_buffer().to_owned();
        drop(ctx);

        assert!(encode_tree_compact(&command_buffer, &buffers, &[13], 14, false).is_none());
        assert!(encode_tree_compact(&command_buffer, &buffers, &[13, 12], 14, false).is_none());
        assert!(encode_tree_compact(&command_buffer, &buffers, &[12, 15], 14, false).is_none());
        // Declared logs must exactly describe the bound LDE buffers.
        assert!(encode_tree_compact(&command_buffer, &buffers, &[11, 12], 14, false).is_none());
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
