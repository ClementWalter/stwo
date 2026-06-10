use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::PackedBaseField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Backend, Col, Column, CpuBackend};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

pub type WideFibonacciComponent<const N: usize> = FrameworkComponent<WideFibonacciEval<N>>;

mod fib_with_preprocessed;

pub struct FibInput {
    pub a: BaseField,
    pub b: BaseField,
}

pub struct FibInputSimd {
    a: PackedBaseField,
    b: PackedBaseField,
}

pub fn generate_trace<const N: usize, B: Backend>(
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    assert!(inputs.len().is_power_of_two());
    let log_size = inputs.len().ilog2();
    let mut trace = (0..N)
        .map(|_| Col::<B, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].set(vec_index, a);
        trace[1].set(vec_index, b);
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.set(vec_index, b);
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<B, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// Same as [`generate_trace`] for the CPU backend, filling row chunks in parallel.
pub fn generate_trace_cpu_parallel<const N: usize>(
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
    assert!(inputs.len().is_power_of_two());
    let log_size = inputs.len().ilog2();
    let mut trace: Vec<Vec<BaseField>> = (0..N)
        .map(|_| Col::<CpuBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();

    // Rows (instances) are independent; fill disjoint row chunks of every column
    // concurrently.
    let chunk_size = 1 << 12;
    let mut col_chunks = trace
        .iter_mut()
        .map(|c| c.chunks_mut(chunk_size))
        .collect_vec();
    let n_chunks = inputs.len().div_ceil(chunk_size);
    let mut chunk_views = (0..n_chunks)
        .map(|i| {
            (
                i * chunk_size,
                col_chunks
                    .iter_mut()
                    .map(|it| it.next().unwrap())
                    .collect_vec(),
            )
        })
        .collect_vec();

    let process_chunk = |(start, cols): &mut (usize, Vec<&mut [BaseField]>)| {
        use stwo::prover::backend::simd::m31::{PackedBaseField, N_LANES};
        let len = cols[0].len();
        // Rows are independent: each SIMD lane carries one instance, so every column
        // write lands as one contiguous cache line instead of 16 scattered scalars.
        let mut idx = 0;
        while idx + N_LANES <= len {
            let mut a =
                PackedBaseField::from_array(std::array::from_fn(|k| inputs[*start + idx + k].a));
            let mut b =
                PackedBaseField::from_array(std::array::from_fn(|k| inputs[*start + idx + k].b));
            cols[0][idx..idx + N_LANES].copy_from_slice(&a.to_array());
            cols[1][idx..idx + N_LANES].copy_from_slice(&b.to_array());
            for col in cols.iter_mut().skip(2) {
                (a, b) = (b, a * a + b * b);
                col[idx..idx + N_LANES].copy_from_slice(&b.to_array());
            }
            idx += N_LANES;
        }
        // Scalar tail for traces smaller than one SIMD vector.
        for idx in idx..len {
            let input = &inputs[*start + idx];
            let mut a = input.a;
            let mut b = input.b;
            cols[0][idx] = a;
            cols[1][idx] = b;
            cols.iter_mut().skip(2).for_each(|col| {
                (a, b) = (b, a.square() + b.square());
                col[idx] = b;
            });
        }
    };

    #[cfg(feature = "parallel")]
    chunk_views.par_iter_mut().for_each(process_chunk);
    #[cfg(not(feature = "parallel"))]
    chunk_views.iter_mut().for_each(process_chunk);

    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// GPU trace generator: one thread per instance computes the whole Fibonacci row and
/// writes each column value into its column buffer (coalesced across threads). Values
/// are bit-identical to [`generate_trace_cpu_parallel`]; returns `None` without a
/// usable device.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn generate_trace_cpu_metal<const N: usize>(
    inputs: &[FibInput],
) -> Option<ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> {
    use std::sync::OnceLock;

    use metal::{
        Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLResourceUsage,
        MTLSize,
    };

    const KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant uint P = 0x7FFFFFFFu;
inline uint m31_add(uint a, uint b) { uint s = a + b; return (s >= P) ? s - P : s; }
inline uint m31_mul(uint a, uint b) {
    ulong p = (ulong)a * (ulong)b;
    uint s = (uint)(p & P) + (uint)(p >> 31);
    s = (s & P) + (s >> 31);
    return (s >= P) ? s - P : s;
}
struct ColPtr { device uint* data; };
kernel void fib_trace(
    device const ColPtr* cols [[buffer(0)]],
    device const uint* a_in [[buffer(1)]],
    device const uint* b_in [[buffer(2)]],
    constant uint& n_rows [[buffer(3)]],
    constant uint& n_cols [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n_rows) { return; }
    uint a = a_in[i];
    uint b = b_in[i];
    cols[0].data[i] = a;
    cols[1].data[i] = b;
    for (uint j = 2; j < n_cols; j++) {
        uint c = m31_add(m31_mul(a, a), m31_mul(b, b));
        cols[j].data[i] = c;
        a = b;
        b = c;
    }
}
"#;

    struct Ctx {
        device: Device,
        queue: CommandQueue,
        pipeline: ComputePipelineState,
    }
    // Metal objects are reference-counted Objective-C handles; uses are serialized by
    // the single-threaded call site.
    unsafe impl Send for Ctx {}
    unsafe impl Sync for Ctx {}

    static CTX: OnceLock<Option<Ctx>> = OnceLock::new();
    let ctx = CTX
        .get_or_init(|| {
            let device = Device::system_default()?;
            if !device.has_unified_memory() {
                return None;
            }
            let library = device
                .new_library_with_source(KERNEL, &metal::CompileOptions::new())
                .ok()?;
            let function = library.get_function("fib_trace", None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            let queue = device.new_command_queue();
            Some(Ctx {
                device,
                queue,
                pipeline,
            })
        })
        .as_ref()?;

    let n_rows = inputs.len();
    let log_size = n_rows.ilog2();
    if n_rows < (1 << 16) {
        return None;
    }
    let (a_in, b_in): (Vec<u32>, Vec<u32>) =
        inputs.iter().map(|input| (input.a.0, input.b.0)).unzip();

    // Safety: the kernel writes every element of every column before anything reads it.
    #[allow(clippy::uninit_vec)]
    let mut trace: Vec<Vec<BaseField>> = (0..N)
        .map(|_| {
            let mut column = Vec::with_capacity(n_rows);
            unsafe { column.set_len(n_rows) };
            column
        })
        .collect_vec();
    // Fault the fresh pages in on the CPU (parallel) so the kernel doesn't stall.
    #[cfg(feature = "parallel")]
    trace.par_iter_mut().for_each(|column| {
        for slot in column.iter_mut().step_by(16384 / 4) {
            *slot = BaseField::from_u32_unchecked(0);
        }
    });

    let bind = |data: &[u32]| -> Buffer {
        let bytes = std::mem::size_of_val(data);
        if (data.as_ptr() as usize).is_multiple_of(16384) && bytes.is_multiple_of(16384) {
            ctx.device.new_buffer_with_bytes_no_copy(
                data.as_ptr() as *const std::ffi::c_void,
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            ctx.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }
    };

    let col_buffers: Vec<Buffer> = trace
        .iter()
        .map(|column| {
            // BaseField is a transparent u32 wrapper.
            let words =
                unsafe { std::slice::from_raw_parts(column.as_ptr() as *const u32, n_rows) };
            bind(words)
        })
        .collect();
    // Columns written zero-copy only: a copied buffer would not land in `trace`.
    if col_buffers.iter().zip(&trace).any(|(buffer, column)| {
        !std::ptr::eq(buffer.contents() as *const BaseField, column.as_ptr())
    }) {
        return None;
    }
    let addresses: Vec<u64> = col_buffers.iter().map(|b| b.gpu_address()).collect();
    let addr_buffer = ctx.device.new_buffer_with_data(
        addresses.as_ptr() as *const std::ffi::c_void,
        (addresses.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let a_buffer = bind(&a_in);
    let b_buffer = bind(&b_in);

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&ctx.pipeline);
    encoder.set_buffer(0, Some(&addr_buffer), 0);
    encoder.set_buffer(1, Some(&a_buffer), 0);
    encoder.set_buffer(2, Some(&b_buffer), 0);
    let rows = n_rows as u32;
    encoder.set_bytes(3, 4, &rows as *const _ as *const std::ffi::c_void);
    let cols = N as u32;
    encoder.set_bytes(4, 4, &cols as *const _ as *const std::ffi::c_void);
    for buffer in &col_buffers {
        encoder.use_resource(buffer, MTLResourceUsage::Write);
    }
    encoder.dispatch_threads(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let domain = CanonicCoset::new(log_size).circle_domain();
    Some(
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, eval))
            .collect_vec(),
    )
}

/// Same as [`generate_trace`] but optimized for simd.
pub fn generate_trace_simd<const N: usize>(
    log_size: u32,
    inputs: &[FibInputSimd],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a separate Fibonacci sequence of length `N`.
#[derive(Clone)]
pub struct WideFibonacciEval<const N: usize> {
    pub log_n_rows: u32,
}
impl<const N: usize> FrameworkEval for WideFibonacciEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }

    /// Same constraints as [`Self::evaluate`], as a Metal body (see the trait docs).
    fn metal_constraint_body(&self) -> Option<String> {
        Some(format!(
            r#"
    uint a = TRACE_AT(1, 0);
    uint b = TRACE_AT(1, 1);
    for (uint j = 2; j < {N}u; j++) {{
        uint c = TRACE_AT(1, j);
        ADD_CONSTRAINT(m31_sub(c, m31_add(m31_mul(a, a), m31_mul(b, b))));
        a = b;
        b = c;
    }}
"#
        ))
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sM31Channel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::channel::Poseidon252Channel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleChannel;
    use stwo::core::verifier::verify;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::{Column, CpuBackend};
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::{prove, CommitmentSchemeProver};
    use stwo_constraint_framework::{
        assert_constraints_on_polys, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
    };

    use super::WideFibonacciEval;
    use crate::wide_fibonacci::{
        generate_trace, generate_trace_cpu_parallel, FibInput, WideFibonacciComponent,
    };

    const FIB_SEQUENCE_LENGTH: usize = 100;

    fn generate_test_inputs(log_n_instances: u32) -> Vec<FibInput> {
        #[cfg(feature = "parallel")]
        return (0..1u32 << log_n_instances)
            .into_par_iter()
            .map(|i| FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i),
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        (0..1 << log_n_instances)
            .map(|i| FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i as u32),
            })
            .collect_vec()
    }

    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { log_n_rows: N }.evaluate(eval);
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let traces = TreeVec::new(vec![
            vec![],
            generate_trace::<FIB_SEQUENCE_LENGTH, SimdBackend>(&generate_test_inputs(
                LOG_N_INSTANCES,
            )),
        ]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test]
    #[should_panic]
    fn test_wide_fibonacci_constraints_fails() {
        const LOG_N_INSTANCES: u32 = 6;

        let mut trace = generate_trace::<FIB_SEQUENCE_LENGTH, SimdBackend>(&generate_test_inputs(
            LOG_N_INSTANCES,
        ));
        // Modify the trace such that a constraint fail.
        trace[17].values.set(2, BaseField::one());
        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test_log::test]
    fn test_wide_fib_prove_with_blake() {
        for log_n_instances in 4..=8 {
            let config = PcsConfig::default();
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    /// Tests the subdomain evaluation path (log_expansion > 0) by using log_blowup_factor = 2
    /// with constraint degree 1, so the committed domain is larger than the eval domain.
    #[test_log::test]
    fn test_wide_fib_prove_with_larger_blowup() {
        for log_n_instances in 4..=7 {
            let config = PcsConfig {
                pow_bits: 10,
                fri_config: FriConfig::new(0, 2, 3, 1),
                lifting_log_size: None,
            };
            // Precompute twiddles for the larger committed domain.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace.
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    /// Same as [test_wide_fib_prove_with_blake] but with FRI fold step > 1.
    #[test]
    fn test_wide_fib_prove_with_blake_with_fri_jumps() {
        for log_n_instances in 4..=8 {
            let mut config = PcsConfig::default();
            // Test different steps.
            config.fri_config.fold_step = if (4..6).contains(&log_n_instances) {
                2
            } else {
                3
            };
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wide_fib_prove_with_poseidon() {
        const LOG_N_INSTANCES: u32 = 6;
        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Poseidon252Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

        // TODO(ilya): remove the following once preprocessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace =
            generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(LOG_N_INSTANCES));
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: LOG_N_INSTANCES,
            },
            SecureField::zero(),
        );
        let proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Poseidon252Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }

    /// End-to-end CPU-backend proof of the wide Fibonacci AIR. The CPU backend is the
    /// proving path for consumers without nightly SIMD; size is overridable for
    /// benchmarking:
    ///   CPU_FIB_LOG_N_INSTANCES=18 cargo test --release test_cpu_e2e_wide_fib_prove
    #[test_log::test]
    fn test_cpu_e2e_wide_fib_prove() {
        let log_n_instances: u32 = std::env::var("CPU_FIB_LOG_N_INSTANCES")
            .map(|s| s.parse().unwrap())
            .unwrap_or(8);
        let config = PcsConfig::default();
        // Twiddle precompute and trace generation are independent; run them in parallel.
        // GPU pipeline compilation is process setup, like twiddle precompute.
        #[cfg(feature = "metal")]
        stwo::prover::backend::metal::warmup();
        let t = std::time::Instant::now();
        let (twiddles, trace) = rayon::join(
            || {
                CpuBackend::precompute_twiddles(
                    CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                        .circle_domain()
                        .half_coset,
                )
            },
            || {
                let inputs = generate_test_inputs(log_n_instances);
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if let Some(trace) = super::generate_trace_cpu_metal::<FIB_SEQUENCE_LENGTH>(&inputs)
                {
                    return trace;
                }
                generate_trace_cpu_parallel::<FIB_SEQUENCE_LENGTH>(&inputs)
            },
        );
        tracing::info!("twiddles + trace gen: {:?}", t.elapsed());

        // Setup protocol.
        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);
        let t = std::time::Instant::now();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);
        tracing::info!("trace commit: {:?}", t.elapsed());

        // Generate component.
        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: log_n_instances,
            },
            SecureField::zero(),
        );

        // Prove.
        let t = std::time::Instant::now();
        let proof = prove::<CpuBackend, Blake2sM31MerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();
        if std::env::var("CPU_FIB_PROOF_HASH").is_ok() {
            use std::hash::{Hash, Hasher};
            let repr = format!("{proof:?}");
            let mut h = std::collections::hash_map::DefaultHasher::new();
            repr.hash(&mut h);
            std::println!("PROOF_HASH={:016x} len={}", h.finish(), repr.len());
        }

        tracing::info!("prove: {:?}", t.elapsed());

        // Verify.
        let t = std::time::Instant::now();
        let verifier_channel = &mut Blake2sM31Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        tracing::info!("verify: {:?}", t.elapsed());
    }

    #[test]
    fn test_e2e_lifted_fib_prove() {
        const LOG_SIZE_SHORT: u32 = 3;
        const LOG_SIZE_LONG: u32 = 9;

        const N_COLS_LONG_COMPONENT: usize = 4;
        const N_COLS_SHORT_COMPONENT: usize = 5;

        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(LOG_SIZE_LONG + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace = [
            generate_trace::<N_COLS_LONG_COMPONENT, _>(&generate_test_inputs(LOG_SIZE_LONG)),
            generate_trace::<N_COLS_SHORT_COMPONENT, _>(&generate_test_inputs(LOG_SIZE_SHORT)),
        ]
        .concat();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Generate components.
        let mut trace_alloc = TraceLocationAllocator::default();
        let component0 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_COLS_LONG_COMPONENT> {
                log_n_rows: LOG_SIZE_LONG,
            },
            SecureField::zero(),
        );
        let component1 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_COLS_SHORT_COMPONENT> {
                log_n_rows: LOG_SIZE_SHORT,
            },
            SecureField::zero(),
        );

        // Prove.
        let proof = prove::<CpuBackend, Blake2sM31MerkleChannel>(
            &[&component0, &component1],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sM31Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

        let trace_sizes = [
            vec![LOG_SIZE_LONG; N_COLS_LONG_COMPONENT],
            vec![LOG_SIZE_SHORT; N_COLS_SHORT_COMPONENT],
        ]
        .concat();
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = TreeVec::new(vec![vec![], trace_sizes]);
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);

        assert!(verify(
            &[&component0, &component1],
            verifier_channel,
            commitment_scheme,
            proof,
        )
        .is_ok());
    }
}
