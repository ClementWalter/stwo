//! Apple-GPU (Metal) kernels accelerating prover hot paths on unified-memory
//! machines. Not a separate proving backend: the CPU/SIMD backends dispatch into
//! these kernels above size thresholds and keep their own implementations as
//! fallbacks (no device, unsupported shape) and as reference-test ground truths.

pub(crate) mod blake2s;
pub mod constraints;
pub(crate) mod fft;
pub(crate) mod quotients;
pub(crate) mod twiddles;

/// Initializes the GPU device and compiles the kernel pipelines (a one-time cost of
/// ~100ms) so the first commitment doesn't pay it. Safe to call from any thread; a
/// no-op when no usable device exists.
pub fn warmup() {
    blake2s::warmup();
    fft::warmup();
    quotients::warmup();
    twiddles::warmup();
}
