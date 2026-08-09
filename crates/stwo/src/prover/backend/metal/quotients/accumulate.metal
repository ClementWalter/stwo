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
    uint n_cols;
    uint mode;
    uint init_acc[4];
    uint coeffs[64];
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
