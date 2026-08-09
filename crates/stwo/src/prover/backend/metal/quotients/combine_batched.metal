#include <metal_stdlib>
using namespace metal;

constant uint P = 0x7FFFFFFFu;
constant uint GROUP_SIZE = 256u;
constant uint MAX_SAMPLES = 6u;

inline uint m31_canonical(uint value) {
    return value == P ? 0u : value;
}

inline uint m31_add(uint a, uint b) {
    uint sum = a + b;
    return sum >= P ? sum - P : sum;
}

inline uint m31_sub(uint a, uint b) {
    return a >= b ? a - b : a + P - b;
}

inline uint m31_mul(uint a, uint b) {
    ulong product = (ulong)a * (ulong)b;
    uint sum = (uint)(product & P) + (uint)(product >> 31);
    sum = (sum & P) + (sum >> 31);
    return sum >= P ? sum - P : sum;
}

// x^(P - 2), where P - 2 = 2^31 - 3 has bits 30..0 set except bit 1.
// Production reaches this once per 256-row threadgroup, never once per sample.
inline uint m31_inverse(uint x) {
    uint result = x;
    for (int bit = 29; bit >= 0; --bit) {
        result = m31_mul(result, result);
        if (bit != 1) { result = m31_mul(result, x); }
    }
    return result;
}

struct CM31 { uint a; uint b; };

inline CM31 canonical_cm31(CM31 value) {
    return CM31{m31_canonical(value.a), m31_canonical(value.b)};
}

inline CM31 cm31_sub(CM31 lhs, CM31 rhs) {
    return CM31{m31_sub(lhs.a, rhs.a), m31_sub(lhs.b, rhs.b)};
}

inline CM31 cm31_mul(CM31 lhs, CM31 rhs) {
    return CM31{
        m31_sub(m31_mul(lhs.a, rhs.a), m31_mul(lhs.b, rhs.b)),
        m31_add(m31_mul(lhs.a, rhs.b), m31_mul(lhs.b, rhs.a)),
    };
}

struct Sample {
    CM31 prx; CM31 pix; CM31 pry; CM31 piy;
    CM31 flt0; CM31 flt1;
    uint log_ratio;
    uint _pad;
};

struct Params {
    uint n_rows;
    uint n_samples;
};

kernel void combine_quotients_batched(
    constant ulong* acc_addrs [[buffer(0)]],
    device const uint* xs [[buffer(1)]],
    device const uint* ys [[buffer(2)]],
    device uint* out0 [[buffer(3)]],
    device uint* out1 [[buffer(4)]],
    device uint* out2 [[buffer(5)]],
    device uint* out3 [[buffer(6)]],
    constant Sample* samples [[buffer(7)]],
    constant Params& p [[buffer(8)]],
    // [global sample mask, first row for sample 0, ..., first row for sample 5].
    device atomic_uint* pole_report [[buffer(9)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    const uint row = group * GROUP_SIZE + lane;
    const bool active = row < p.n_rows;
    const uint x = active ? m31_canonical(xs[row]) : 0u;
    const uint y = active ? m31_canonical(ys[row]) : 0u;

    uint denominators_a[MAX_SAMPLES];
    uint denominators_b[MAX_SAMPLES];
    uint norms[MAX_SAMPLES];
    uint sample_prefixes[MAX_SAMPLES];
    uint row_product = 1u;
    uint pole_mask = 0u;

    // The fixed-size register arrays avoid per-row dynamic memory. Every field word is
    // canonicalized at its load boundary, including raw P representations of zero.
    for (uint sample = 0; sample < p.n_samples; ++sample) {
        constant Sample& raw = samples[sample];
        const CM31 prx = canonical_cm31(raw.prx);
        const CM31 pix = canonical_cm31(raw.pix);
        const CM31 pry = canonical_cm31(raw.pry);
        const CM31 piy = canonical_cm31(raw.piy);
        const CM31 dx = CM31{m31_sub(prx.a, x), prx.b};
        const CM31 dy = CM31{m31_sub(pry.a, y), pry.b};
        const CM31 denominator = cm31_sub(cm31_mul(dx, piy), cm31_mul(dy, pix));
        const uint norm = m31_add(
            m31_mul(denominator.a, denominator.a),
            m31_mul(denominator.b, denominator.b));

        denominators_a[sample] = denominator.a;
        denominators_b[sample] = denominator.b;
        // Keep the Montgomery product invertible even on a pole. The command still
        // completes uniformly and the host rejects its outputs using `pole_report`.
        const uint safe_norm = norm == 0u ? 1u : norm;
        norms[sample] = safe_norm;
        sample_prefixes[sample] = row_product;
        row_product = m31_mul(row_product, safe_norm);
        if (norm == 0u) { pole_mask |= 1u << sample; }
    }
    if (!active) { row_product = 1u; }

    // Two 256-word arrays are exactly 2 KiB of threadgroup storage. They hold the
    // exclusive products before and after each lane. Tail lanes contribute identity,
    // and all 256 lanes cross every barrier uniformly.
    threadgroup uint prefixes[GROUP_SIZE];
    threadgroup uint suffixes[GROUP_SIZE];
    prefixes[lane] = row_product;
    suffixes[GROUP_SIZE - 1u - lane] = row_product;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < GROUP_SIZE; stride <<= 1u) {
        const uint index = (lane + 1u) * (stride << 1u) - 1u;
        if (index < GROUP_SIZE) {
            prefixes[index] = m31_mul(prefixes[index - stride], prefixes[index]);
            suffixes[index] = m31_mul(suffixes[index - stride], suffixes[index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup uint group_product[1];
    if (lane == GROUP_SIZE - 1u) {
        group_product[0] = prefixes[GROUP_SIZE - 1u];
        prefixes[GROUP_SIZE - 1u] = 1u;
        suffixes[GROUP_SIZE - 1u] = 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GROUP_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        const uint index = (lane + 1u) * (stride << 1u) - 1u;
        if (index < GROUP_SIZE) {
            const uint prefix_left = prefixes[index - stride];
            const uint prefix_parent = prefixes[index];
            prefixes[index - stride] = prefix_parent;
            prefixes[index] = m31_mul(prefix_parent, prefix_left);

            const uint suffix_left = suffixes[index - stride];
            const uint suffix_parent = suffixes[index];
            suffixes[index - stride] = suffix_parent;
            suffixes[index] = m31_mul(suffix_parent, suffix_left);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup uint group_product_inverse[1];
    if (lane == 0u) { group_product_inverse[0] = m31_inverse(group_product[0]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_product_inverse = m31_mul(
        m31_mul(prefixes[lane], suffixes[GROUP_SIZE - 1u - lane]),
        group_product_inverse[0]);
    // Recover each norm inverse in reverse Montgomery order. The resulting array is
    // consumed forward below, preserving the CPU's canonical sample accumulation order.
    for (int sample = int(p.n_samples) - 1; sample >= 0; --sample) {
        const uint norm = norms[sample];
        norms[sample] = m31_mul(row_product_inverse, sample_prefixes[sample]);
        row_product_inverse = m31_mul(row_product_inverse, norm);
    }

    if (active && pole_mask != 0u) {
        atomic_fetch_or_explicit(&pole_report[0], pole_mask, memory_order_relaxed);
        for (uint sample = 0u; sample < p.n_samples; ++sample) {
            if ((pole_mask & (1u << sample)) != 0u) {
                atomic_fetch_min_explicit(&pole_report[1u + sample], row, memory_order_relaxed);
            }
        }
    }

    if (!active) { return; }

    CM31 quotient0 = CM31{0u, 0u};
    CM31 quotient1 = CM31{0u, 0u};
    for (uint sample = 0u; sample < p.n_samples; ++sample) {
        constant Sample& raw = samples[sample];
        const uint inverse_norm = norms[sample];
        const CM31 inverse = CM31{
            m31_mul(denominators_a[sample], inverse_norm),
            m31_mul(m31_sub(0u, denominators_b[sample]), inverse_norm),
        };
        const uint log_ratio = raw.log_ratio;
        const uint lifted = ((row >> (log_ratio + 1u)) << 1u) | (row & 1u);
        device const uint* acc0 = (device const uint*)acc_addrs[sample * 4u + 0u];
        device const uint* acc1 = (device const uint*)acc_addrs[sample * 4u + 1u];
        device const uint* acc2 = (device const uint*)acc_addrs[sample * 4u + 2u];
        device const uint* acc3 = (device const uint*)acc_addrs[sample * 4u + 3u];
        const CM31 flt0 = canonical_cm31(raw.flt0);
        const CM31 flt1 = canonical_cm31(raw.flt1);
        CM31 numerator0 = CM31{
            m31_sub(m31_canonical(acc0[lifted]), m31_mul(flt0.a, y)),
            m31_sub(m31_canonical(acc1[lifted]), m31_mul(flt0.b, y)),
        };
        CM31 numerator1 = CM31{
            m31_sub(m31_canonical(acc2[lifted]), m31_mul(flt1.a, y)),
            m31_sub(m31_canonical(acc3[lifted]), m31_mul(flt1.b, y)),
        };
        numerator0 = cm31_mul(numerator0, inverse);
        numerator1 = cm31_mul(numerator1, inverse);
        quotient0 = CM31{
            m31_add(quotient0.a, numerator0.a),
            m31_add(quotient0.b, numerator0.b),
        };
        quotient1 = CM31{
            m31_add(quotient1.a, numerator1.a),
            m31_add(quotient1.b, numerator1.b),
        };
    }
    out0[row] = quotient0.a;
    out1[row] = quotient0.b;
    out2[row] = quotient1.a;
    out3[row] = quotient1.b;
}
