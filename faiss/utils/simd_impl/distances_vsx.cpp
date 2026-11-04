/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>
#include <faiss/utils/simd_impl/masked_sum.h>

#include <cmath>
#include <limits>
#include <vector>

#define THE_SIMD_LEVEL SIMDLevel::VSX
#include <faiss/impl/simdlib/simdlib_ppc64.h>
#include <faiss/utils/simd_impl/distances_autovec-inl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

#include <faiss/utils/simd_impl/distances_simdlib256.h>

namespace faiss {

namespace {

using namespace ppc64_vsx;

inline float vsx_hsum(ppc_f32x4 v) {
    float buf[4];
    st(buf, v);
    return (buf[0] + buf[1]) + (buf[2] + buf[3]);
}

inline size_t vsx_argmin(const float* dis, size_t ny) {
    ppc_f32x4 vminv = vec_splats(HUGE_VALF);
    ppc_u32x4 iminv = vec_splats(0u);
    const ppc_u32x4 incv = vec_splats(4u);
    static const uint32_t iota_arr[4] = {0, 1, 2, 3};
    ppc_u32x4 iv = ld(iota_arr);

    size_t i = 0;
    for (; i + 3 < ny; i += 4) {
        const ppc_f32x4 d = ld(dis + i);
        const ppc_u32x4 lt = (ppc_u32x4)vec_cmplt(d, vminv);
        vminv = vec_min(d, vminv);
        iminv = vec_sel(iminv, iv, lt);
        iv = vec_add(iv, incv);
    }

    float vmin_buf[4];
    uint32_t imin_buf[4];
    st(vmin_buf, vminv);
    st(reinterpret_cast<uint8_t*>(imin_buf), (ppc_u8x16)iminv);

    float min_dis = vmin_buf[0];
    size_t nearest_idx = imin_buf[0];
    for (int k = 1; k < 4; ++k) {
        if (vmin_buf[k] < min_dis) {
            min_dis = vmin_buf[k];
            nearest_idx = imin_buf[k];
        }
    }

    for (; i < ny; ++i) {
        if (dis[i] < min_dis) {
            min_dis = dis[i];
            nearest_idx = i;
        }
    }
    return nearest_idx;
}

} // anonymous namespace

template <>
void fvec_madd<SIMDLevel::VSX>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    const ppc_f32x4 bfv = vec_splats(bf);
    size_t i = 0;

    // Process 16 floats (four 128-bit VSX vectors) at a time
    for (; i + 15 < n; i += 16) {
        const ppc_f32x4 a0 = ld(a + i);
        const ppc_f32x4 a1 = ld(a + i + 4);
        const ppc_f32x4 a2 = ld(a + i + 8);
        const ppc_f32x4 a3 = ld(a + i + 12);

        const ppc_f32x4 b0 = ld(b + i);
        const ppc_f32x4 b1 = ld(b + i + 4);
        const ppc_f32x4 b2 = ld(b + i + 8);
        const ppc_f32x4 b3 = ld(b + i + 12);

        st(c + i, vec_madd(bfv, b0, a0));
        st(c + i + 4, vec_madd(bfv, b1, a1));
        st(c + i + 8, vec_madd(bfv, b2, a2));
        st(c + i + 12, vec_madd(bfv, b3, a3));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        const ppc_f32x4 ai = ld(a + i);
        const ppc_f32x4 bi = ld(b + i);
        st(c + i, vec_madd(bfv, bi, ai));
    }

    // Scalar remainder
    for (; i < n; ++i) {
        c[i] = a[i] + bf * b[i];
    }
}

template <>
float fvec_masked_sum<SIMDLevel::VSX>(
        const float* x,
        const uint8_t* bits,
        size_t d) {
    static const uint32_t mask_low_arr[4] = {1, 2, 4, 8};
    static const uint32_t mask_high_arr[4] = {16, 32, 64, 128};
    const ppc_u32x4 mask_low = ld(mask_low_arr);
    const ppc_u32x4 mask_high = ld(mask_high_arr);

    ppc_f32x4 acc0 = vec_splats(0.0f);
    ppc_f32x4 acc1 = vec_splats(0.0f);

    const size_t full_bytes = d / 8;
    for (size_t b = 0; b < full_bytes; ++b) {
        const ppc_u32x4 b_bcast = vec_splats(static_cast<uint32_t>(bits[b]));
        const ppc_u32x4 m0 = (ppc_u32x4)vec_cmpeq(
                vec_and(b_bcast, mask_low), mask_low);
        const ppc_u32x4 m1 = (ppc_u32x4)vec_cmpeq(
                vec_and(b_bcast, mask_high), mask_high);

        const ppc_f32x4 v0 = ld(x + b * 8);
        const ppc_f32x4 v1 = ld(x + b * 8 + 4);

        acc0 = vec_add(acc0, (ppc_f32x4)vec_and((ppc_u32x4)v0, m0));
        acc1 = vec_add(acc1, (ppc_f32x4)vec_and((ppc_u32x4)v1, m1));
    }

    float sum = vsx_hsum(vec_add(acc0, acc1));

    // Scalar remainder for d % 8
    for (size_t j = full_bytes * 8; j < d; ++j) {
        if ((bits[j / 8] >> (j % 8)) & 1) {
            sum += x[j];
        }
    }
    return sum;
}

template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::VSX>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    float x_sqlen = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        x_sqlen += x[j] * x[j];
    }
    const ppc_f32x4 vx_sqlen = vec_splats(x_sqlen);

    // Pre-broadcast -2 * x[j] outside the ny loop
    ppc_f32x4 m2_xj_buf[64];
    std::vector<ppc_f32x4> m2_xj_vec;
    const ppc_f32x4* m2_xj = m2_xj_buf;
    if (d > 64) {
        m2_xj_vec.resize(d);
        for (size_t j = 0; j < d; ++j) {
            m2_xj_vec[j] = vec_splats(-2.0f * x[j]);
        }
        m2_xj = m2_xj_vec.data();
    } else {
        for (size_t j = 0; j < d; ++j) {
            m2_xj_buf[j] = vec_splats(-2.0f * x[j]);
        }
    }

    size_t i = 0;
    // Process 8 vectors at a time
    for (; i + 7 < ny; i += 8) {
        ppc_f32x4 dp0 = vx_sqlen;
        ppc_f32x4 dp1 = vx_sqlen;

        for (size_t j = 0; j < d; ++j) {
            const ppc_f32x4 y0 = ld(y + i + j * d_offset);
            const ppc_f32x4 y1 = ld(y + i + 4 + j * d_offset);
            dp0 = vec_madd(m2_xj[j], y0, dp0);
            dp1 = vec_madd(m2_xj[j], y1, dp1);
        }

        const ppc_f32x4 ysq0 = ld(y_sqlen + i);
        const ppc_f32x4 ysq1 = ld(y_sqlen + i + 4);
        st(dis + i, vec_add(dp0, ysq0));
        st(dis + i + 4, vec_add(dp1, ysq1));
    }

    // Process 4 vectors at a time
    for (; i + 3 < ny; i += 4) {
        ppc_f32x4 dp = vx_sqlen;
        for (size_t j = 0; j < d; ++j) {
            const ppc_f32x4 y0 = ld(y + i + j * d_offset);
            dp = vec_madd(m2_xj[j], y0, dp);
        }
        const ppc_f32x4 ysq = ld(y_sqlen + i);
        st(dis + i, vec_add(dp, ysq));
    }

    // Scalar cleanup
    for (; i < ny; ++i) {
        float dp = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            dp += x[j] * y[i + j * d_offset];
        }
        dis[i] = x_sqlen + y_sqlen[i] - 2.0f * dp;
    }
}

template <>
void fvec_inner_products_ny<SIMDLevel::VSX>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    // Process 4 vectors of y at a time in parallel
    for (; i + 3 < ny; i += 4) {
        const float* y0 = y;
        const float* y1 = y + d;
        const float* y2 = y + 2 * d;
        const float* y3 = y + 3 * d;

        ppc_f32x4 acc0 = vec_splats(0.0f);
        ppc_f32x4 acc1 = vec_splats(0.0f);
        ppc_f32x4 acc2 = vec_splats(0.0f);
        ppc_f32x4 acc3 = vec_splats(0.0f);

        size_t j = 0;
        for (; j + 3 < d; j += 4) {
            const ppc_f32x4 vx = ld(x + j);
            acc0 = vec_madd(vx, ld(y0 + j), acc0);
            acc1 = vec_madd(vx, ld(y1 + j), acc1);
            acc2 = vec_madd(vx, ld(y2 + j), acc2);
            acc3 = vec_madd(vx, ld(y3 + j), acc3);
        }

        float s0 = vsx_hsum(acc0);
        float s1 = vsx_hsum(acc1);
        float s2 = vsx_hsum(acc2);
        float s3 = vsx_hsum(acc3);

        for (; j < d; ++j) {
            const float xj = x[j];
            s0 += xj * y0[j];
            s1 += xj * y1[j];
            s2 += xj * y2[j];
            s3 += xj * y3[j];
        }

        dis[i] = s0;
        dis[i + 1] = s1;
        dis[i + 2] = s2;
        dis[i + 3] = s3;

        y += 4 * d;
    }

    // Process remaining vectors one at a time
    for (; i < ny; ++i) {
        ppc_f32x4 acc = vec_splats(0.0f);
        size_t j = 0;
        for (; j + 3 < d; j += 4) {
            acc = vec_madd(ld(x + j), ld(y + j), acc);
        }
        float s = vsx_hsum(acc);
        for (; j < d; ++j) {
            s += x[j] * y[j];
        }
        dis[i] = s;
        y += d;
    }
}

template <>
void fvec_L2sqr_ny<SIMDLevel::VSX>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    // Process 4 vectors of y at a time in parallel
    for (; i + 3 < ny; i += 4) {
        const float* y0 = y;
        const float* y1 = y + d;
        const float* y2 = y + 2 * d;
        const float* y3 = y + 3 * d;

        ppc_f32x4 acc0 = vec_splats(0.0f);
        ppc_f32x4 acc1 = vec_splats(0.0f);
        ppc_f32x4 acc2 = vec_splats(0.0f);
        ppc_f32x4 acc3 = vec_splats(0.0f);

        size_t j = 0;
        for (; j + 3 < d; j += 4) {
            const ppc_f32x4 vx = ld(x + j);
            const ppc_f32x4 diff0 = vec_sub(vx, ld(y0 + j));
            const ppc_f32x4 diff1 = vec_sub(vx, ld(y1 + j));
            const ppc_f32x4 diff2 = vec_sub(vx, ld(y2 + j));
            const ppc_f32x4 diff3 = vec_sub(vx, ld(y3 + j));

            acc0 = vec_madd(diff0, diff0, acc0);
            acc1 = vec_madd(diff1, diff1, acc1);
            acc2 = vec_madd(diff2, diff2, acc2);
            acc3 = vec_madd(diff3, diff3, acc3);
        }

        float s0 = vsx_hsum(acc0);
        float s1 = vsx_hsum(acc1);
        float s2 = vsx_hsum(acc2);
        float s3 = vsx_hsum(acc3);

        for (; j < d; ++j) {
            const float q0 = x[j] - y0[j];
            const float q1 = x[j] - y1[j];
            const float q2 = x[j] - y2[j];
            const float q3 = x[j] - y3[j];
            s0 += q0 * q0;
            s1 += q1 * q1;
            s2 += q2 * q2;
            s3 += q3 * q3;
        }

        dis[i] = s0;
        dis[i + 1] = s1;
        dis[i + 2] = s2;
        dis[i + 3] = s3;

        y += 4 * d;
    }

    // Process remaining vectors one at a time
    for (; i < ny; ++i) {
        ppc_f32x4 acc = vec_splats(0.0f);
        size_t j = 0;
        for (; j + 3 < d; j += 4) {
            const ppc_f32x4 diff = vec_sub(ld(x + j), ld(y + j));
            acc = vec_madd(diff, diff, acc);
        }
        float s = vsx_hsum(acc);
        for (; j < d; ++j) {
            const float diff = x[j] - y[j];
            s += diff * diff;
        }
        dis[i] = s;
        y += d;
    }
}

template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::VSX>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny<SIMDLevel::VSX>(distances_tmp_buffer, x, y, d, ny);
    return vsx_argmin(distances_tmp_buffer, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::VSX>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    fvec_L2sqr_ny_transposed<SIMDLevel::VSX>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
    return vsx_argmin(distances_tmp_buffer, ny);
}

template <>
int fvec_madd_and_argmin<SIMDLevel::VSX>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    const ppc_f32x4 bfv = vec_splats(bf);

    ppc_f32x4 vminv = vec_splats(1e20f);
    ppc_u32x4 iminv = vec_splats(0u);
    const ppc_u32x4 incv = vec_splats(4u);
    static const uint32_t iota_arr[4] = {0, 1, 2, 3};
    ppc_u32x4 iv = ld(iota_arr);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        const ppc_f32x4 ai = ld(a + i);
        const ppc_f32x4 bi = ld(b + i);
        const ppc_f32x4 ci = vec_madd(bfv, bi, ai);
        st(c + i, ci);

        const ppc_u32x4 less_than = (ppc_u32x4)vec_cmplt(ci, vminv);
        vminv = vec_min(ci, vminv);
        iminv = vec_sel(iminv, iv, less_than);
        iv = vec_add(iv, incv);
    }

    float vmin_buf[4];
    uint32_t imin_buf[4];
    st(vmin_buf, vminv);
    st(reinterpret_cast<uint8_t*>(imin_buf), (ppc_u8x16)iminv);

    float vmin = vmin_buf[0];
    int imin = static_cast<int>(imin_buf[0]);
    for (int k = 1; k < 4; ++k) {
        if (vmin_buf[k] < vmin) {
            vmin = vmin_buf[k];
            imin = static_cast<int>(imin_buf[k]);
        }
    }

    for (; i < n; ++i) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = static_cast<int>(i);
        }
    }
    return imin;
}

} // namespace faiss
