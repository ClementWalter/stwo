//! Apple-GPU out-of-domain sampling: the dot product of each polynomial's
//! coefficients with the shared FFT basis column.
//!
//! One threadgroup per polynomial: threads accumulate strided partial sums of
//! `coeff * basis` per QM31 coordinate (coefficients are M31, so each coordinate is an
//! independent M31 dot), then tree-reduce in threadgroup memory. Field addition is
//! associative and commutative, so the regrouped sum equals the CPU fold exactly.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};

use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
use crate::core::constraints::{coset_vanishing, coset_vanishing_derivative};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::prover::backend::CpuBackend;
use crate::prover::poly::circle::{BarycentricEvalGroup, CircleCoefficients, CircleEvaluation};
use crate::prover::poly::BitReversedOrder;

/// Minimum coefficient count for GPU dispatch.
pub(crate) const MIN_METAL_OOD_LOG_SIZE: u32 = 16;
/// Evaluation-form barycentric batches need enough independent columns to occupy
/// the GPU and amortize command/buffer setup over the parallel CPU reference path.
const MIN_METAL_BARYCENTRIC_COLUMNS: usize = 64;
/// Measured crossover floor in M31 x QM31 row-products (2^21 rows x 64 columns).
const MIN_METAL_BARYCENTRIC_WORK: usize = 1 << 27;

struct BarycentricPointTables {
    bases: Vec<[u32; 2]>,
    offsets: [[u32; 2]; 128],
}

/// Splits each bit-reversed 256-row block into one block base and 128
/// lane-pair offsets. Even/odd lanes share an underlying point and select its
/// first/conjugate-half representative respectively.
fn barycentric_point_tables(log_size: u32) -> BarycentricPointTables {
    assert!(log_size >= 8);
    let half_coset = CanonicCoset::new(log_size).circle_domain().half_coset;
    let block_log = log_size - 8;
    let bases = (0..1 << block_log)
        .map(|block| {
            let point = half_coset.at(bit_reverse_index(block, block_log));
            [point.x.0, point.y.0]
        })
        .collect();
    let offsets = std::array::from_fn(|lane_pair| {
        let multiplier = bit_reverse_index(lane_pair, 7) << block_log;
        let point = half_coset.step.mul(multiplier as u128);
        [point.x.0, point.y.0]
    });
    BarycentricPointTables { bases, offsets }
}

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

inline uint m31_sub(uint a, uint b) {
    return (a >= b) ? a - b : a + P - b;
}

inline uint m31_neg(uint a) {
    return a == 0u ? 0u : P - a;
}

inline uint m31_inv(uint x) {
    uint r = x;
    for (int i = 29; i >= 0; i--) {
        r = m31_mul(r, r);
        if (i != 1) { r = m31_mul(r, x); }
    }
    return r;
}

struct Cm31 { uint a; uint b; };
struct Qm31 { uint a; uint b; uint c; uint d; };
struct Point { uint x; uint y; };

inline Cm31 cm_add(Cm31 x, Cm31 y) {
    return Cm31{m31_add(x.a, y.a), m31_add(x.b, y.b)};
}

inline Cm31 cm_sub(Cm31 x, Cm31 y) {
    return Cm31{m31_sub(x.a, y.a), m31_sub(x.b, y.b)};
}

inline Cm31 cm_neg(Cm31 x) {
    return Cm31{m31_neg(x.a), m31_neg(x.b)};
}

inline Cm31 cm_mul(Cm31 x, Cm31 y) {
    return Cm31{
        m31_sub(m31_mul(x.a, y.a), m31_mul(x.b, y.b)),
        m31_add(m31_mul(x.a, y.b), m31_mul(x.b, y.a))
    };
}

inline Cm31 cm_inv(Cm31 x) {
    uint norm_inv = m31_inv(m31_add(m31_mul(x.a, x.a), m31_mul(x.b, x.b)));
    return Cm31{m31_mul(x.a, norm_inv), m31_neg(m31_mul(x.b, norm_inv))};
}

inline Qm31 qm_add(Qm31 x, Qm31 y) {
    return Qm31{m31_add(x.a, y.a), m31_add(x.b, y.b),
                m31_add(x.c, y.c), m31_add(x.d, y.d)};
}

inline Qm31 qm_sub(Qm31 x, Qm31 y) {
    return Qm31{m31_sub(x.a, y.a), m31_sub(x.b, y.b),
                m31_sub(x.c, y.c), m31_sub(x.d, y.d)};
}

inline Qm31 qm_neg(Qm31 x) {
    return Qm31{m31_neg(x.a), m31_neg(x.b), m31_neg(x.c), m31_neg(x.d)};
}

inline Qm31 qm_mul_m31(Qm31 x, uint y) {
    return Qm31{m31_mul(x.a, y), m31_mul(x.b, y),
                m31_mul(x.c, y), m31_mul(x.d, y)};
}

inline Qm31 qm_mul(Qm31 x, Qm31 y) {
    Cm31 x0 = Cm31{x.a, x.b};
    Cm31 x1 = Cm31{x.c, x.d};
    Cm31 y0 = Cm31{y.a, y.b};
    Cm31 y1 = Cm31{y.c, y.d};
    Cm31 x1y1 = cm_mul(x1, y1);
    Cm31 r_x1y1 = cm_mul(Cm31{2u, 1u}, x1y1);
    Cm31 z0 = cm_add(cm_mul(x0, y0), r_x1y1);
    Cm31 z1 = cm_add(cm_mul(x0, y1), cm_mul(x1, y0));
    return Qm31{z0.a, z0.b, z1.a, z1.b};
}

inline Qm31 qm_inv(Qm31 x) {
    Cm31 x0 = Cm31{x.a, x.b};
    Cm31 x1 = Cm31{x.c, x.d};
    Cm31 x1_sq = cm_mul(x1, x1);
    Cm31 i_x1_sq = Cm31{m31_neg(x1_sq.b), x1_sq.a};
    Cm31 denom = cm_sub(cm_mul(x0, x0), cm_add(cm_add(x1_sq, x1_sq), i_x1_sq));
    Cm31 denom_inv = cm_inv(denom);
    Cm31 z0 = cm_mul(x0, denom_inv);
    Cm31 z1 = cm_neg(cm_mul(x1, denom_inv));
    return Qm31{z0.a, z0.b, z1.a, z1.b};
}

inline Point point_add(Point x, Point y) {
    return Point{
        m31_sub(m31_mul(x.x, y.x), m31_mul(x.y, y.y)),
        m31_add(m31_mul(x.x, y.y), m31_mul(x.y, y.x))
    };
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

struct BarycentricParams {
    uint px_a; uint px_b; uint px_c; uint px_d;
    uint py_a; uint py_b; uint py_c; uint py_d;
    uint common_a; uint common_b; uint common_c; uint common_d;
    uint n_rows;
};

kernel void barycentric_weights(
    device const Point* bases [[buffer(0)]],
    device const Point* offsets [[buffer(1)]],
    device Qm31* out [[buffer(2)]],
    constant BarycentricParams& p [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]])
{
    uint i = (block << 8u) + lane;
    if (i >= p.n_rows) { return; }
    Point q = point_add(bases[block], offsets[lane >> 1u]);
    if ((lane & 1u) != 0u) { q.y = m31_neg(q.y); }

    Qm31 px = Qm31{p.px_a, p.px_b, p.px_c, p.px_d};
    Qm31 py = Qm31{p.py_a, p.py_b, p.py_c, p.py_d};
    // Circle-group subtraction h = p - q, not coordinate-wise subtraction.
    Qm31 hx = qm_add(qm_mul_m31(px, q.x), qm_mul_m31(py, q.y));
    Qm31 hy = qm_sub(qm_mul_m31(py, q.x), qm_mul_m31(px, q.y));
    Qm31 numerator = qm_add(Qm31{1u, 0u, 0u, 0u}, hx);
    Qm31 vi_inverse = qm_mul(numerator, qm_inv(hy));
    Qm31 common = Qm31{p.common_a, p.common_b, p.common_c, p.common_d};
    Qm31 weight = qm_mul(vi_inverse, common);
    out[i] = (i & 1u) == 0u ? weight : qm_neg(weight);
}
"#;

struct OodContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    barycentric_pipeline: ComputePipelineState,
    point_tables: HashMap<u32, (Buffer, Buffer)>,
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
            let barycentric_function = library.get_function("barycentric_weights", None).ok()?;
            let barycentric_pipeline = device
                .new_compute_pipeline_state_with_function(&barycentric_function)
                .ok()?;
            let queue = shared.queue.clone();
            Some(Mutex::new(OodContext {
                device,
                queue,
                pipeline,
                barycentric_pipeline,
                point_tables: HashMap::new(),
            }))
        })
        .as_ref()
}

/// Forces device/pipeline initialization; see [`super::warmup`].
pub(crate) fn warmup() {
    let _ = context();
}

/// Returns whether the static out-of-domain pipeline compiled successfully.
pub(crate) fn is_ready() -> bool {
    let Some(context) = context() else {
        return false;
    };
    context.lock().is_ok()
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

fn point_table_buffers(ctx: &mut OodContext, log_size: u32) -> (Buffer, Buffer) {
    if !ctx.point_tables.contains_key(&log_size) {
        let tables = barycentric_point_tables(log_size);
        let bases = ctx.device.new_buffer_with_data(
            tables.bases.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(tables.bases.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let offsets = ctx.device.new_buffer_with_data(
            tables.offsets.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&tables.offsets) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        ctx.point_tables.insert(log_size, (bases, offsets));
    }
    let (bases, offsets) = &ctx.point_tables[&log_size];
    (bases.clone(), offsets.clone())
}

fn barycentric_common(coset: CanonicCoset, p: CirclePoint<SecureField>) -> Option<SecureField> {
    let domain = coset.circle_domain();
    let p0 = domain.at(0).into_ef::<SecureField>();
    let si0 = SecureField::from(1u32)
        / ((p0.y * SecureField::from(-2))
            * coset_vanishing_derivative(
                Coset::new(CirclePointIndex::generator(), domain.log_size()),
                p0,
            ));
    let vn = coset_vanishing(coset.coset, p);
    // The full canonic domain is closed under the half-turn. Therefore V_D(p) != 0
    // excludes both p = q and p = antipode(q), the h = (-1, 0) pole admitted by
    // the rational point-vanishing inverse.
    (vn != SecureField::default()).then_some(si0 * vn)
}

#[repr(C)]
struct BarycentricParams {
    px: [u32; 4],
    py: [u32; 4],
    common: [u32; 4],
    n_rows: u32,
}

/// Test-only resident generation of one barycentric-weight column. The output is
/// separate from every committed evaluation, so a failed command remains safe for
/// the caller to discard and recompute on CPU before transcript mutation.
#[cfg(test)]
pub(crate) fn barycentric_weights_metal(
    coset: CanonicCoset,
    p: CirclePoint<SecureField>,
) -> Option<Vec<SecureField>> {
    let log_size = coset.log_size();
    if log_size < MIN_METAL_OOD_LOG_SIZE {
        return None;
    }
    let n_rows = 1usize << log_size;
    let common = barycentric_common(coset, p)?;
    let ctx = context()?;
    let mut ctx = ctx.lock().unwrap();
    let out_buffer = ctx.device.new_buffer(
        (n_rows * std::mem::size_of::<[u32; 4]>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (bases, offsets) = point_table_buffers(&mut ctx, log_size);
    let params = BarycentricParams {
        px: p.x.to_m31_array().map(|value| value.0),
        py: p.y.to_m31_array().map(|value| value.0),
        common: common.to_m31_array().map(|value| value.0),
        n_rows: n_rows as u32,
    };

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&ctx.barycentric_pipeline);
    encoder.set_buffer(0, Some(&bases), 0);
    encoder.set_buffer(1, Some(&offsets), 0);
    encoder.set_buffer(2, Some(&out_buffer), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<BarycentricParams>() as u64,
        &params as *const _ as *const std::ffi::c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new((n_rows >> 8) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    super::context::wait_for_completion(command_buffer).ok()?;
    // SAFETY: `out_buffer` is a live StorageModeShared allocation of exactly
    // `n_rows * 4` u32 words. Successful checked completion initialized the entire
    // allocation, Metal contents are u32-aligned, and the immutable slice is consumed
    // synchronously before the buffer handle or locked context can be dropped or mutated.
    // The Metal ABI is explicit raw coordinates; do not depend on the nested Rust
    // QM31/CM31 layout, which intentionally has no repr(C) guarantee.
    let words =
        unsafe { std::slice::from_raw_parts(out_buffer.contents() as *const u32, n_rows * 4) };
    Some(
        words
            .chunks_exact(4)
            .map(|coordinates| {
                SecureField::from_m31_array(std::array::from_fn(|coordinate| {
                    BaseField::from_u32_unchecked(coordinates[coordinate])
                }))
            })
            .collect(),
    )
}

struct PreparedBarycentricGroup {
    group_index: usize,
    n_rows: usize,
    n_columns: usize,
    bases: Buffer,
    offsets: Buffer,
    column_buffers: Vec<Buffer>,
    address_buffer: Buffer,
    output_buffer: Buffer,
    params: BarycentricParams,
}

/// Generates every eligible group's weights and immediately consumes them in
/// grouped dot products using one private scratch allocation, one command buffer,
/// and one checked wait. Ineligible groups and any failed submission remain `None`
/// for CPU/SIMD fallback before the transcript observes sampled values.
pub(crate) fn barycentric_eval_groups_metal(
    groups: &[BarycentricEvalGroup<'_, CpuBackend>],
) -> Vec<Option<Vec<SecureField>>> {
    let mut results: Vec<Option<Vec<SecureField>>> = groups.iter().map(|_| None).collect();
    let Some(ctx) = context() else {
        return results;
    };
    let mut ctx = ctx.lock().unwrap();
    let mut prepared = Vec::new();
    let mut max_rows = 0usize;

    for (group_index, group) in groups.iter().enumerate() {
        let log_size = group.coset.log_size();
        if log_size < MIN_METAL_OOD_LOG_SIZE || group.evals.is_empty() {
            continue;
        }
        let n_rows = 1usize << log_size;
        let expected_domain = group.coset.circle_domain();
        if group.evals.iter().any(|evaluation| {
            evaluation.values.len() != n_rows || evaluation.domain != expected_domain
        }) {
            continue;
        }
        let Some(common) = barycentric_common(group.coset, group.point) else {
            continue;
        };
        let (bases, offsets) = point_table_buffers(&mut ctx, log_size);
        let mut column_buffers = Vec::with_capacity(group.evals.len());
        let mut addresses = Vec::with_capacity(group.evals.len());
        for evaluation in &group.evals {
            let buffer = bind_input(
                &ctx.device,
                evaluation.values.as_ptr() as *const u8,
                n_rows * 4,
            );
            addresses.push(buffer.gpu_address());
            column_buffers.push(buffer);
        }
        let address_buffer = ctx.device.new_buffer_with_data(
            addresses.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(addresses.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = ctx.device.new_buffer(
            (group.evals.len() * 16) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        prepared.push(PreparedBarycentricGroup {
            group_index,
            n_rows,
            n_columns: group.evals.len(),
            bases,
            offsets,
            column_buffers,
            address_buffer,
            output_buffer,
            params: BarycentricParams {
                px: group.point.x.to_m31_array().map(|value| value.0),
                py: group.point.y.to_m31_array().map(|value| value.0),
                common: common.to_m31_array().map(|value| value.0),
                n_rows: n_rows as u32,
            },
        });
        max_rows = max_rows.max(n_rows);
    }
    if prepared.is_empty() {
        return results;
    }

    let scratch = ctx.device.new_buffer(
        (max_rows * 16) as u64,
        MTLResourceOptions::StorageModePrivate,
    );
    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    for group in &prepared {
        encoder.set_compute_pipeline_state(&ctx.barycentric_pipeline);
        encoder.set_buffer(0, Some(&group.bases), 0);
        encoder.set_buffer(1, Some(&group.offsets), 0);
        encoder.set_buffer(2, Some(&scratch), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<BarycentricParams>() as u64,
            &group.params as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new((group.n_rows >> 8) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.memory_barrier_with_resources(&[scratch.as_ref()]);

        encoder.set_compute_pipeline_state(&ctx.pipeline);
        encoder.set_buffer(0, Some(&group.address_buffer), 0);
        encoder.set_buffer(1, Some(&scratch), 0);
        encoder.set_buffer(2, Some(&group.output_buffer), 0);
        let rows = group.n_rows as u32;
        encoder.set_bytes(3, 4, &rows as *const _ as *const std::ffi::c_void);
        for buffer in &group.column_buffers {
            encoder.use_resource(buffer, MTLResourceUsage::Read);
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(group.n_columns as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.memory_barrier_with_resources(&[scratch.as_ref()]);
    }
    encoder.end_encoding();
    command_buffer.commit();
    if super::context::wait_for_completion(command_buffer).is_err() {
        return results;
    }

    for group in prepared {
        // SAFETY: each live StorageModeShared output buffer contains exactly
        // `n_columns * 4` u32 words. Successful checked completion initialized every word,
        // Metal contents are u32-aligned, and the locked context prevents concurrent command
        // mutation while this immutable slice is synchronously converted into owned values.
        let words = unsafe {
            std::slice::from_raw_parts(
                group.output_buffer.contents() as *const u32,
                group.n_columns * 4,
            )
        };
        results[group.group_index] = Some(
            words
                .chunks_exact(4)
                .map(|coordinates| {
                    SecureField::from_m31_array(std::array::from_fn(|coordinate| {
                        BaseField::from_u32_unchecked(coordinates[coordinate])
                    }))
                })
                .collect(),
        );
    }
    results
}

/// GPU shared-basis evaluation of many same-size polynomials at one point; equals
/// mapping the CPU dot exactly (summation regrouping only). Returns `None` without a
/// usable device or successful command.
fn eval_m31_columns_metal(
    columns: &[&[BaseField]],
    weights: &[SecureField],
) -> Option<Vec<SecureField>> {
    let n_rows = weights.len();
    if n_rows < (1 << MIN_METAL_OOD_LOG_SIZE) || columns.is_empty() {
        return None;
    }
    let ctx = context()?;
    let ctx = ctx.lock().unwrap();

    let mut col_buffers = Vec::with_capacity(columns.len());
    let mut addresses: Vec<u64> = Vec::with_capacity(columns.len());
    for column in columns {
        assert_eq!(column.len(), n_rows);
        let buffer = bind_input(&ctx.device, column.as_ptr() as *const u8, n_rows * 4);
        addresses.push(buffer.gpu_address());
        col_buffers.push(buffer);
    }
    let addr_buffer = ctx.device.new_buffer_with_data(
        addresses.as_ptr() as *const std::ffi::c_void,
        (addresses.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let packed_weights: Vec<[u32; 4]> = weights
        .iter()
        .map(|weight| weight.to_m31_array().map(|coordinate| coordinate.0))
        .collect();
    let basis_buffer = bind_input(
        &ctx.device,
        packed_weights.as_ptr() as *const u8,
        std::mem::size_of_val(packed_weights.as_slice()),
    );
    let out_buffer = ctx.device.new_buffer(
        (columns.len() * 16) as u64,
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
        MTLSize::new(columns.len() as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    super::context::wait_for_completion(command_buffer).ok()?;

    // SAFETY: `out_buffer` remains a live StorageModeShared allocation of exactly
    // `columns.len() * 4` u32 words. Successful checked completion initialized every word,
    // Metal contents are u32-aligned, and the locked context prevents concurrent mutation
    // while the immutable slice is synchronously converted into owned values.
    let words = unsafe {
        std::slice::from_raw_parts(out_buffer.contents() as *const u32, columns.len() * 4)
    };
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

/// GPU shared-basis evaluation of many same-size coefficient columns at one point.
pub(crate) fn eval_many_metal(
    polys: &[&CircleCoefficients<CpuBackend>],
    basis: &[SecureField],
) -> Option<Vec<SecureField>> {
    let columns: Vec<&[BaseField]> = polys.iter().map(|poly| poly.coeffs.as_slice()).collect();
    eval_m31_columns_metal(&columns, basis)
}

/// GPU shared-weight barycentric evaluation of many same-domain evaluation columns at
/// one point. This is the same M31 x QM31 dot product as coefficient evaluation; only
/// the source column's representation differs.
pub(crate) fn barycentric_eval_many_metal(
    evals: &[&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>],
    weights: &[SecureField],
) -> Option<Vec<SecureField>> {
    if !should_batch_barycentric_on_metal(weights.len(), evals.len()) {
        return None;
    }
    let columns: Vec<&[BaseField]> = evals
        .iter()
        .map(|evaluation| evaluation.values.as_slice())
        .collect();
    eval_m31_columns_metal(&columns, weights)
}

const fn should_batch_barycentric_on_metal(n_rows: usize, n_columns: usize) -> bool {
    n_columns >= MIN_METAL_BARYCENTRIC_COLUMNS
        && n_rows.saturating_mul(n_columns) >= MIN_METAL_BARYCENTRIC_WORK
}

#[cfg(test)]
#[path = "ood_tests.rs"]
mod tests;
