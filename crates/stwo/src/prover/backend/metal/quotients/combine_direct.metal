#include <metal_stdlib>
using namespace metal;

constant uint P = 0x7FFFFFFFu;

inline uint canonical(uint value) { return value == P ? 0u : value; }

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

struct CM31 { uint a; uint b; };

inline CM31 canon_cm31(CM31 value) {
    return CM31{canonical(value.a), canonical(value.b)};
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

// Independent test-only control: one CM31 Fermat inverse per row and sample.
inline CM31 cm31_inverse(CM31 x) {
    CM31 result = x;
    for (int bit = 60; bit >= 0; --bit) {
        result = cm31_mul(result, result);
        if (bit != 32) { result = cm31_mul(result, x); }
    }
    return result;
}

struct Sample {
    CM31 prx; CM31 pix; CM31 pry; CM31 piy;
    CM31 flt0; CM31 flt1;
    uint log_ratio;
    uint _pad;
};

struct Params { uint n_rows; uint n_samples; };

kernel void combine_quotients_direct(
    constant ulong* acc_addrs [[buffer(0)]],
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
    const uint x = canonical(xs[row]);
    const uint y = canonical(ys[row]);
    CM31 quotient0 = CM31{0u, 0u};
    CM31 quotient1 = CM31{0u, 0u};
    for (uint sample = 0u; sample < p.n_samples; ++sample) {
        constant Sample& raw = samples[sample];
        const CM31 prx = canon_cm31(raw.prx);
        const CM31 pix = canon_cm31(raw.pix);
        const CM31 pry = canon_cm31(raw.pry);
        const CM31 piy = canon_cm31(raw.piy);
        const CM31 dx = CM31{m31_sub(prx.a, x), prx.b};
        const CM31 dy = CM31{m31_sub(pry.a, y), pry.b};
        const CM31 inverse = cm31_inverse(cm31_sub(cm31_mul(dx, piy), cm31_mul(dy, pix)));

        const uint lifted = ((row >> (raw.log_ratio + 1u)) << 1u) | (row & 1u);
        device const uint* acc0 = (device const uint*)acc_addrs[sample * 4u + 0u];
        device const uint* acc1 = (device const uint*)acc_addrs[sample * 4u + 1u];
        device const uint* acc2 = (device const uint*)acc_addrs[sample * 4u + 2u];
        device const uint* acc3 = (device const uint*)acc_addrs[sample * 4u + 3u];
        const CM31 flt0 = canon_cm31(raw.flt0);
        const CM31 flt1 = canon_cm31(raw.flt1);
        CM31 numerator0 = CM31{
            m31_sub(canonical(acc0[lifted]), m31_mul(flt0.a, y)),
            m31_sub(canonical(acc1[lifted]), m31_mul(flt0.b, y)),
        };
        CM31 numerator1 = CM31{
            m31_sub(canonical(acc2[lifted]), m31_mul(flt1.a, y)),
            m31_sub(canonical(acc3[lifted]), m31_mul(flt1.b, y)),
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
