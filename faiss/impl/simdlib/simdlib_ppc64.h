/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <faiss/impl/simdlib/simdlib.h>

// On little-endian POWER with VSX (the default for POWER8 and later with
// ppc64le toolchains) the operations below are implemented with 128-bit
// VSX/AltiVec vectors. A 256-bit simdlib register is handled as two 128-bit
// halves, which are the "lanes" referred to in the comments below.

#if defined(__PPC64__) && defined(__VSX__) && defined(__BYTE_ORDER__) && \
        (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define FAISS_SIMDLIB_PPC64_VSX 1
#include <altivec.h>
// altivec.h may define these as context-sensitive keyword macros, which
// conflicts with std::vector and bool in C++ code.
#undef vector
#undef pixel
#undef bool
#endif

namespace faiss {

#ifdef FAISS_SIMDLIB_PPC64_VSX

namespace ppc64_vsx {

using ppc_u8x16 = __vector unsigned char;
using ppc_s8x16 = __vector signed char;
using ppc_u16x8 = __vector unsigned short;
using ppc_u32x4 = __vector unsigned int;
using ppc_f32x4 = __vector float;

// Unaligned 128-bit loads / stores, in memory (element) order.
inline ppc_u8x16 ld(const uint8_t* p) {
    return vec_xl(0, p);
}
inline ppc_u16x8 ld(const uint16_t* p) {
    return vec_xl(0, p);
}
inline ppc_u32x4 ld(const uint32_t* p) {
    return vec_xl(0, p);
}
inline ppc_f32x4 ld(const float* p) {
    return vec_xl(0, p);
}
inline void st(uint8_t* p, ppc_u8x16 v) {
    vec_xst(v, 0, p);
}
inline void st(uint16_t* p, ppc_u16x8 v) {
    vec_xst(v, 0, p);
}
inline void st(uint32_t* p, ppc_u32x4 v) {
    vec_xst(v, 0, p);
}
inline void st(float* p, ppc_f32x4 v) {
    vec_xst(v, 0, p);
}

// Bit i of the result is set iff byte i of m is 0xff (bytes must be 0x00 or
// 0xff). Equivalent of _mm_movemask_epi8.
inline uint32_t movemask16(ppc_u8x16 m) {
    const ppc_u8x16 weights = {
            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    // word k = sum of the (weighted) bytes 4k .. 4k + 3
    const ppc_u32x4 s = vec_sum4s(vec_and(m, weights), vec_splats(0u));
    uint32_t t[4];
    st(t, s);
    return (t[0] + t[1]) | ((t[2] + t[3]) << 8);
}

inline uint32_t movemask32(ppc_u8x16 m0, ppc_u8x16 m1) {
    return movemask16(m0) | (movemask16(m1) << 16);
}

// byte permutation controls selecting the even / odd 32-bit words
// (resp. 16-bit halfwords) of the concatenation (a, b)
inline ppc_u8x16 perm_even32() {
    const ppc_u8x16 p = {0, 1, 2,  3,  8,  9,  10, 11,
                         16, 17, 18, 19, 24, 25, 26, 27};
    return p;
}
inline ppc_u8x16 perm_odd32() {
    const ppc_u8x16 p = {4,  5,  6,  7,  12, 13, 14, 15,
                         20, 21, 22, 23, 28, 29, 30, 31};
    return p;
}
inline ppc_u8x16 perm_even16() {
    const ppc_u8x16 p = {0,  1,  4,  5,  8,  9,  12, 13,
                         16, 17, 20, 21, 24, 25, 28, 29};
    return p;
}
inline ppc_u8x16 perm_odd16() {
    const ppc_u8x16 p = {2,  3,  6,  7,  10, 11, 14, 15,
                         18, 19, 22, 23, 26, 27, 30, 31};
    return p;
}

} // namespace ppc64_vsx

template <>
struct simd256bit_tpl<SIMDLevel::VSX> {
    union {
        uint8_t u8[32];
        uint16_t u16[16];
        uint32_t u32[8];
        float f32[8];
    };

    simd256bit_tpl() {}

    explicit simd256bit_tpl(const void* x) {
        memcpy(u8, x, 32);
    }

    void clear() {
        memset(u8, 0, 32);
    }

    void storeu(void* ptr) const {
        memcpy(ptr, u8, 32);
    }

    void loadu(const void* ptr) {
        memcpy(u8, ptr, 32);
    }

    void store(void* ptr) const {
        storeu(ptr);
    }

    void bin(char bits[257]) const {
        const char* bytes = (char*)this->u8;
        for (int i = 0; i < 256; i++) {
            bits[i] = '0' + ((bytes[i / 8] >> (i % 8)) & 1);
        }
        bits[256] = 0;
    }

    std::string bin() const {
        char bits[257];
        bin(bits);
        return std::string(bits);
    }

    // Checks whether the other holds exactly the same bytes.
    bool is_same_as(simd256bit_tpl other) const {
        using namespace ppc64_vsx;
        return vec_all_eq(ld(u32), ld(other.u32)) &&
                vec_all_eq(ld(u32 + 4), ld(other.u32 + 4));
    }
};

/// vector of 16 elements in uint16
template <>
struct simd16uint16_tpl<SIMDLevel::VSX>
        : simd256bit_tpl<SIMDLevel::VSX> {
    simd16uint16_tpl() {}

    explicit simd16uint16_tpl(int x) {
        set1(x);
    }

    explicit simd16uint16_tpl(uint16_t x) {
        set1(x);
    }

    explicit simd16uint16_tpl(const simd256bit_tpl<SIMDLevel::VSX>& x)
            : simd256bit_tpl<SIMDLevel::VSX>(x) {}

    explicit simd16uint16_tpl(const uint16_t* x)
            : simd256bit_tpl<SIMDLevel::VSX>((const void*)x) {}

    explicit simd16uint16_tpl(
            uint16_t u0,
            uint16_t u1,
            uint16_t u2,
            uint16_t u3,
            uint16_t u4,
            uint16_t u5,
            uint16_t u6,
            uint16_t u7,
            uint16_t u8,
            uint16_t u9,
            uint16_t u10,
            uint16_t u11,
            uint16_t u12,
            uint16_t u13,
            uint16_t u14,
            uint16_t u15) {
        this->u16[0] = u0;
        this->u16[1] = u1;
        this->u16[2] = u2;
        this->u16[3] = u3;
        this->u16[4] = u4;
        this->u16[5] = u5;
        this->u16[6] = u6;
        this->u16[7] = u7;
        this->u16[8] = u8;
        this->u16[9] = u9;
        this->u16[10] = u10;
        this->u16[11] = u11;
        this->u16[12] = u12;
        this->u16[13] = u13;
        this->u16[14] = u14;
        this->u16[15] = u15;
    }

    std::string elements_to_string(const char* fmt) const {
        char res[1000], *ptr = res;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
        for (int i = 0; i < 16; i++) {
            ptr += snprintf(
                    ptr, (size_t)(res + sizeof(res) - ptr), fmt, u16[i]);
        }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }

    std::string dec() const {
        return elements_to_string("%3d,");
    }

    void set1(uint16_t x) {
        using namespace ppc64_vsx;
        const ppc_u16x8 v = vec_splats(x);
        st(u16, v);
        st(u16 + 8, v);
    }

    simd16uint16_tpl operator*(const simd16uint16_tpl& other) const {
        using namespace ppc64_vsx;
        const ppc_u16x8 zero = vec_splats((uint16_t)0);
        simd16uint16_tpl c;
        // low 16 bits of a * b + 0
        st(c.u16, vec_mladd(ld(u16), ld(other.u16), zero));
        st(c.u16 + 8, vec_mladd(ld(u16 + 8), ld(other.u16 + 8), zero));
        return c;
    }

    // shift must be known at compile time (and in [0, 15])
    simd16uint16_tpl operator>>(const int shift) const {
        using namespace ppc64_vsx;
        const ppc_u16x8 s = vec_splats((uint16_t)shift);
        simd16uint16_tpl c;
        st(c.u16, vec_sr(ld(u16), s));
        st(c.u16 + 8, vec_sr(ld(u16 + 8), s));
        return c;
    }

    // shift must be known at compile time (and in [0, 15])
    simd16uint16_tpl operator<<(const int shift) const {
        using namespace ppc64_vsx;
        const ppc_u16x8 s = vec_splats((uint16_t)shift);
        simd16uint16_tpl c;
        st(c.u16, vec_sl(ld(u16), s));
        st(c.u16 + 8, vec_sl(ld(u16 + 8), s));
        return c;
    }

    simd16uint16_tpl& operator+=(const simd16uint16_tpl& other) {
        *this = *this + other;
        return *this;
    }

    simd16uint16_tpl& operator-=(const simd16uint16_tpl& other) {
        *this = *this - other;
        return *this;
    }

    simd16uint16_tpl operator+(const simd16uint16_tpl& other) const {
        using namespace ppc64_vsx;
        simd16uint16_tpl c;
        st(c.u16, vec_add(ld(u16), ld(other.u16)));
        st(c.u16 + 8, vec_add(ld(u16 + 8), ld(other.u16 + 8)));
        return c;
    }

    simd16uint16_tpl operator-(const simd16uint16_tpl& other) const {
        using namespace ppc64_vsx;
        simd16uint16_tpl c;
        st(c.u16, vec_sub(ld(u16), ld(other.u16)));
        st(c.u16 + 8, vec_sub(ld(u16 + 8), ld(other.u16 + 8)));
        return c;
    }

    simd16uint16_tpl operator&(
            const simd256bit_tpl<SIMDLevel::VSX>& other) const {
        using namespace ppc64_vsx;
        simd16uint16_tpl c;
        st(c.u8, vec_and(ld(u8), ld(other.u8)));
        st(c.u8 + 16, vec_and(ld(u8 + 16), ld(other.u8 + 16)));
        return c;
    }

    simd16uint16_tpl operator|(
            const simd256bit_tpl<SIMDLevel::VSX>& other) const {
        using namespace ppc64_vsx;
        simd16uint16_tpl c;
        st(c.u8, vec_or(ld(u8), ld(other.u8)));
        st(c.u8 + 16, vec_or(ld(u8 + 16), ld(other.u8 + 16)));
        return c;
    }

    simd16uint16_tpl operator^(
            const simd256bit_tpl<SIMDLevel::VSX>& other) const {
        using namespace ppc64_vsx;
        simd16uint16_tpl c;
        st(c.u8, vec_xor(ld(u8), ld(other.u8)));
        st(c.u8 + 16, vec_xor(ld(u8 + 16), ld(other.u8 + 16)));
        return c;
    }

    // returns binary masks
    simd16uint16_tpl operator==(const simd16uint16_tpl& other) const {
        using namespace ppc64_vsx;
        simd16uint16_tpl c;
        st(c.u16, (ppc_u16x8)vec_cmpeq(ld(u16), ld(other.u16)));
        st(c.u16 + 8, (ppc_u16x8)vec_cmpeq(ld(u16 + 8), ld(other.u16 + 8)));
        return c;
    }

    simd16uint16_tpl operator~() const {
        using namespace ppc64_vsx;
        const ppc_u8x16 a0 = ld(u8);
        const ppc_u8x16 a1 = ld(u8 + 16);
        simd16uint16_tpl c;
        st(c.u8, vec_nor(a0, a0));
        st(c.u8 + 16, vec_nor(a1, a1));
        return c;
    }

    // get scalar at index 0
    uint16_t get_scalar_0() const {
        return u16[0];
    }

    // mask of elements where this >= thresh
    // 2 bit per component: 16 * 2 = 32 bit
    uint32_t ge_mask(const simd16uint16_tpl& thresh) const {
        using namespace ppc64_vsx;
        // a >= t  <=>  max(a, t) == a. Each 16-bit element of the comparison
        // result is 0x0000 or 0xffff, so the byte movemask gives exactly 2
        // bits per component.
        const ppc_u16x8 a0 = ld(u16);
        const ppc_u16x8 a1 = ld(u16 + 8);
        const ppc_u16x8 m0 =
                (ppc_u16x8)vec_cmpeq(vec_max(a0, ld(thresh.u16)), a0);
        const ppc_u16x8 m1 =
                (ppc_u16x8)vec_cmpeq(vec_max(a1, ld(thresh.u16 + 8)), a1);
        return movemask32((ppc_u8x16)m0, (ppc_u8x16)m1);
    }

    uint32_t le_mask(const simd16uint16_tpl& thresh) const {
        return thresh.ge_mask(*this);
    }

    uint32_t gt_mask(const simd16uint16_tpl& thresh) const {
        return ~le_mask(thresh);
    }

    bool all_gt(const simd16uint16_tpl& thresh) const {
        using namespace ppc64_vsx;
        return vec_all_gt(ld(u16), ld(thresh.u16)) &&
                vec_all_gt(ld(u16 + 8), ld(thresh.u16 + 8));
    }

    // for debugging only
    uint16_t operator[](int i) const {
        return u16[i];
    }

    void accu_min(const simd16uint16_tpl& incoming) {
        using namespace ppc64_vsx;
        st(u16, vec_min(ld(u16), ld(incoming.u16)));
        st(u16 + 8, vec_min(ld(u16 + 8), ld(incoming.u16 + 8)));
    }

    void accu_max(const simd16uint16_tpl& incoming) {
        using namespace ppc64_vsx;
        st(u16, vec_max(ld(u16), ld(incoming.u16)));
        st(u16 + 8, vec_max(ld(u16 + 8), ld(incoming.u16 + 8)));
    }
};

// not really a std::min because it returns an elementwise min
inline simd16uint16_tpl<SIMDLevel::VSX> min(
        const simd16uint16_tpl<SIMDLevel::VSX>& av,
        const simd16uint16_tpl<SIMDLevel::VSX>& bv) {
    using namespace ppc64_vsx;
    simd16uint16_tpl<SIMDLevel::VSX> c;
    st(c.u16, vec_min(ld(av.u16), ld(bv.u16)));
    st(c.u16 + 8, vec_min(ld(av.u16 + 8), ld(bv.u16 + 8)));
    return c;
}

inline simd16uint16_tpl<SIMDLevel::VSX> max(
        const simd16uint16_tpl<SIMDLevel::VSX>& av,
        const simd16uint16_tpl<SIMDLevel::VSX>& bv) {
    using namespace ppc64_vsx;
    simd16uint16_tpl<SIMDLevel::VSX> c;
    st(c.u16, vec_max(ld(av.u16), ld(bv.u16)));
    st(c.u16 + 8, vec_max(ld(av.u16 + 8), ld(bv.u16 + 8)));
    return c;
}

// decompose in 128-lanes: a = (a0, a1), b = (b0, b1)
// return (a0 + a1, b0 + b1)
inline simd16uint16_tpl<SIMDLevel::VSX> combine2x2(
        const simd16uint16_tpl<SIMDLevel::VSX>& a,
        const simd16uint16_tpl<SIMDLevel::VSX>& b) {
    simd16uint16_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    st(c.u16, vec_add(ld(a.u16), ld(a.u16 + 8)));
    st(c.u16 + 8, vec_add(ld(b.u16), ld(b.u16 + 8)));
    return c;
}

// compare d0 and d1 to thr, return 32 bits corresponding to the concatenation
// of d0 and d1 with thr
inline uint32_t cmp_ge32(
        const simd16uint16_tpl<SIMDLevel::VSX>& d0,
        const simd16uint16_tpl<SIMDLevel::VSX>& d1,
        const simd16uint16_tpl<SIMDLevel::VSX>& thr) {
    using namespace ppc64_vsx;
    const ppc_u16x8 t0 = ld(thr.u16);
    const ppc_u16x8 t1 = ld(thr.u16 + 8);
    const ppc_u16x8 a0 = ld(d0.u16);
    const ppc_u16x8 a1 = ld(d0.u16 + 8);
    const ppc_u16x8 b0 = ld(d1.u16);
    const ppc_u16x8 b1 = ld(d1.u16 + 8);
    // x >= t  <=>  max(x, t) == x; vec_pack truncates the 0x0000 / 0xffff
    // results to 0x00 / 0xff bytes, in element order
    const ppc_u8x16 ma = vec_pack(
            (ppc_u16x8)vec_cmpeq(vec_max(a0, t0), a0),
            (ppc_u16x8)vec_cmpeq(vec_max(a1, t1), a1));
    const ppc_u8x16 mb = vec_pack(
            (ppc_u16x8)vec_cmpeq(vec_max(b0, t0), b0),
            (ppc_u16x8)vec_cmpeq(vec_max(b1, t1), b1));
    return movemask32(ma, mb);
}

inline uint32_t cmp_le32(
        const simd16uint16_tpl<SIMDLevel::VSX>& d0,
        const simd16uint16_tpl<SIMDLevel::VSX>& d1,
        const simd16uint16_tpl<SIMDLevel::VSX>& thr) {
    using namespace ppc64_vsx;
    const ppc_u16x8 t0 = ld(thr.u16);
    const ppc_u16x8 t1 = ld(thr.u16 + 8);
    const ppc_u16x8 a0 = ld(d0.u16);
    const ppc_u16x8 a1 = ld(d0.u16 + 8);
    const ppc_u16x8 b0 = ld(d1.u16);
    const ppc_u16x8 b1 = ld(d1.u16 + 8);
    // x <= t  <=>  min(x, t) == x
    const ppc_u8x16 ma = vec_pack(
            (ppc_u16x8)vec_cmpeq(vec_min(a0, t0), a0),
            (ppc_u16x8)vec_cmpeq(vec_min(a1, t1), a1));
    const ppc_u8x16 mb = vec_pack(
            (ppc_u16x8)vec_cmpeq(vec_min(b0, t0), b0),
            (ppc_u16x8)vec_cmpeq(vec_min(b1, t1), b1));
    return movemask32(ma, mb);
}

// hadd does not cross lanes
inline simd16uint16_tpl<SIMDLevel::VSX> hadd(
        const simd16uint16_tpl<SIMDLevel::VSX>& a,
        const simd16uint16_tpl<SIMDLevel::VSX>& b) {
    simd16uint16_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    const ppc_u8x16 pe = perm_even16();
    const ppc_u8x16 po = perm_odd16();
    for (int lane = 0; lane < 2; lane++) {
        const ppc_u16x8 al = ld(a.u16 + 8 * lane);
        const ppc_u16x8 bl = ld(b.u16 + 8 * lane);
        st(c.u16 + 8 * lane,
           vec_add(vec_perm(al, bl, pe), vec_perm(al, bl, po)));
    }
    return c;
}

// Vectorized version of the following code:
//   for (size_t i = 0; i < n; i++) {
//      bool flag = (candidateValues[i] < currentValues[i]);
//      minValues[i] = flag ? candidateValues[i] : currentValues[i];
//      minIndices[i] = flag ? candidateIndices[i] : currentIndices[i];
//      maxValues[i] = !flag ? candidateValues[i] : currentValues[i];
//      maxIndices[i] = !flag ? candidateIndices[i] : currentIndices[i];
//   }
// Max indices evaluation is inaccurate in case of equal values (the index of
// the last equal value is saved instead of the first one), but this behavior
// saves instructions.
inline void cmplt_min_max_fast(
        const simd16uint16_tpl<SIMDLevel::VSX> candidateValues,
        const simd16uint16_tpl<SIMDLevel::VSX> candidateIndices,
        const simd16uint16_tpl<SIMDLevel::VSX> currentValues,
        const simd16uint16_tpl<SIMDLevel::VSX> currentIndices,
        simd16uint16_tpl<SIMDLevel::VSX>& minValues,
        simd16uint16_tpl<SIMDLevel::VSX>& minIndices,
        simd16uint16_tpl<SIMDLevel::VSX>& maxValues,
        simd16uint16_tpl<SIMDLevel::VSX>& maxIndices) {
    using namespace ppc64_vsx;
    for (int lane = 0; lane < 2; lane++) {
        const int o = 8 * lane;
        const ppc_u16x8 cv = ld(candidateValues.u16 + o);
        const ppc_u16x8 ci = ld(candidateIndices.u16 + o);
        const ppc_u16x8 uv = ld(currentValues.u16 + o);
        const ppc_u16x8 ui = ld(currentIndices.u16 + o);
        // flag mask: candidate < current
        const ppc_u16x8 flag = (ppc_u16x8)vec_cmplt(cv, uv);
        st(minValues.u16 + o, vec_sel(uv, cv, flag));
        st(minIndices.u16 + o, vec_sel(ui, ci, flag));
        st(maxValues.u16 + o, vec_sel(cv, uv, flag));
        st(maxIndices.u16 + o, vec_sel(ci, ui, flag));
    }
}

// vector of 32 unsigned 8-bit integers
template <>
struct simd32uint8_tpl<SIMDLevel::VSX>
        : simd256bit_tpl<SIMDLevel::VSX> {
    simd32uint8_tpl() {}

    explicit simd32uint8_tpl(int x) {
        set1(x);
    }

    explicit simd32uint8_tpl(uint8_t x) {
        set1(x);
    }
    template <
            uint8_t _0,
            uint8_t _1,
            uint8_t _2,
            uint8_t _3,
            uint8_t _4,
            uint8_t _5,
            uint8_t _6,
            uint8_t _7,
            uint8_t _8,
            uint8_t _9,
            uint8_t _10,
            uint8_t _11,
            uint8_t _12,
            uint8_t _13,
            uint8_t _14,
            uint8_t _15,
            uint8_t _16,
            uint8_t _17,
            uint8_t _18,
            uint8_t _19,
            uint8_t _20,
            uint8_t _21,
            uint8_t _22,
            uint8_t _23,
            uint8_t _24,
            uint8_t _25,
            uint8_t _26,
            uint8_t _27,
            uint8_t _28,
            uint8_t _29,
            uint8_t _30,
            uint8_t _31>
    void set() {
        const uint8_t x[32] = {
                _0,  _1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9,  _10,
                _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21,
                _22, _23, _24, _25, _26, _27, _28, _29, _30, _31};
        loadu(x);
    }

    explicit simd32uint8_tpl(const simd256bit_tpl<SIMDLevel::VSX>& x)
            : simd256bit_tpl<SIMDLevel::VSX>(x) {}

    explicit simd32uint8_tpl(const uint8_t* x)
            : simd256bit_tpl<SIMDLevel::VSX>((const void*)x) {}

    std::string elements_to_string(const char* fmt) const {
        char res[1000], *ptr = res;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
        for (int i = 0; i < 32; i++) {
            ptr += snprintf(ptr, (size_t)(res + sizeof(res) - ptr), fmt, u8[i]);
        }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%02x,");
    }

    std::string dec() const {
        return elements_to_string("%3d,");
    }

    void set1(uint8_t x) {
        using namespace ppc64_vsx;
        const ppc_u8x16 v = vec_splats(x);
        st(u8, v);
        st(u8 + 16, v);
    }

    simd32uint8_tpl operator&(
            const simd256bit_tpl<SIMDLevel::VSX>& other) const {
        using namespace ppc64_vsx;
        simd32uint8_tpl c;
        st(c.u8, vec_and(ld(u8), ld(other.u8)));
        st(c.u8 + 16, vec_and(ld(u8 + 16), ld(other.u8 + 16)));
        return c;
    }

    simd32uint8_tpl operator+(const simd32uint8_tpl& other) const {
        using namespace ppc64_vsx;
        simd32uint8_tpl c;
        st(c.u8, vec_add(ld(u8), ld(other.u8)));
        st(c.u8 + 16, vec_add(ld(u8 + 16), ld(other.u8 + 16)));
        return c;
    }

    simd32uint8_tpl& operator+=(const simd32uint8_tpl& other) {
        *this = *this + other;
        return *this;
    }

    simd32uint8_tpl operator-(const simd32uint8_tpl& other) const {
        using namespace ppc64_vsx;
        simd32uint8_tpl c;
        st(c.u8, vec_sub(ld(u8), ld(other.u8)));
        st(c.u8 + 16, vec_sub(ld(u8 + 16), ld(other.u8 + 16)));
        return c;
    }

    simd32uint8_tpl& operator-=(const simd32uint8_tpl& other) {
        *this = *this - other;
        return *this;
    }

    uint8_t operator[](int i) const {
        return u8[i];
    }

    // The very important operation that everything relies on
    simd32uint8_tpl lookup_2_lanes(const simd32uint8_tpl& idx) const {
        simd32uint8_tpl c;
        // Each 16-byte lane is a table; vperm does the lookup in a single
        // instruction. Only the low bits of the control bytes are used by
        // vperm, so they are masked to 0..15 and bytes with the MSB set are
        // zeroed afterwards.
        using namespace ppc64_vsx;
        const ppc_u8x16 low_nibble = vec_splats((uint8_t)0x0f);
        const ppc_s8x16 zero = vec_splats((signed char)0);
        for (int lane = 0; lane < 2; lane++) {
            const ppc_u8x16 table = ld(u8 + 16 * lane);
            const ppc_u8x16 ix = ld(idx.u8 + 16 * lane);
            const ppc_u8x16 looked_up =
                    vec_perm(table, table, vec_and(ix, low_nibble));
            const ppc_u8x16 msb_set = (ppc_u8x16)vec_cmplt((ppc_s8x16)ix, zero);
            st(c.u8 + 16 * lane, vec_andc(looked_up, msb_set));
        }
        return c;
    }
};

inline simd32uint8_tpl<SIMDLevel::VSX> uint16_to_uint8_saturate(
        const simd16uint16_tpl<SIMDLevel::VSX>& a,
        const simd16uint16_tpl<SIMDLevel::VSX>& b) {
    simd32uint8_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    // lane 0: (sat(a0), sat(b0)), lane 1: (sat(a1), sat(b1))
    st(c.u8, vec_packs(ld(a.u16), ld(b.u16)));
    st(c.u8 + 16, vec_packs(ld(a.u16 + 8), ld(b.u16 + 8)));
    return c;
}

/// get most significant bit of each byte
inline uint32_t get_MSBs(const simd32uint8_tpl<SIMDLevel::VSX>& a) {
    using namespace ppc64_vsx;
    const ppc_s8x16 zero = vec_splats((signed char)0);
    return movemask32(
            (ppc_u8x16)vec_cmplt((ppc_s8x16)ld(a.u8), zero),
            (ppc_u8x16)vec_cmplt((ppc_s8x16)ld(a.u8 + 16), zero));
}

/// use MSB of each byte of mask to select a byte between a and b
inline simd32uint8_tpl<SIMDLevel::VSX> blendv(
        const simd32uint8_tpl<SIMDLevel::VSX>& a,
        const simd32uint8_tpl<SIMDLevel::VSX>& b,
        const simd32uint8_tpl<SIMDLevel::VSX>& mask) {
    simd32uint8_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    const ppc_s8x16 zero = vec_splats((signed char)0);
    for (int lane = 0; lane < 2; lane++) {
        const int o = 16 * lane;
        const ppc_u8x16 m =
                (ppc_u8x16)vec_cmplt((ppc_s8x16)ld(mask.u8 + o), zero);
        st(c.u8 + o, vec_sel(ld(a.u8 + o), ld(b.u8 + o), m));
    }
    return c;
}

// vector of 8 unsigned 32-bit integers
template <>
struct simd8uint32_tpl<SIMDLevel::VSX>
        : simd256bit_tpl<SIMDLevel::VSX> {
    simd8uint32_tpl() {}

    explicit simd8uint32_tpl(uint32_t x) {
        set1(x);
    }

    explicit simd8uint32_tpl(const simd256bit_tpl<SIMDLevel::VSX>& x)
            : simd256bit_tpl<SIMDLevel::VSX>(x) {}

    explicit simd8uint32_tpl(const uint32_t* x)
            : simd256bit_tpl<SIMDLevel::VSX>((const void*)x) {}

    explicit simd8uint32_tpl(
            uint32_t u0,
            uint32_t u1,
            uint32_t u2,
            uint32_t u3,
            uint32_t u4,
            uint32_t u5,
            uint32_t u6,
            uint32_t u7) {
        this->u32[0] = u0;
        this->u32[1] = u1;
        this->u32[2] = u2;
        this->u32[3] = u3;
        this->u32[4] = u4;
        this->u32[5] = u5;
        this->u32[6] = u6;
        this->u32[7] = u7;
    }

    std::string elements_to_string(const char* fmt) const {
        char res[1000], *ptr = res;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
        for (int i = 0; i < 8; i++) {
            ptr += snprintf(
                    ptr, (size_t)(res + sizeof(res) - ptr), fmt, u32[i]);
        }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }

    std::string hex() const {
        return elements_to_string("%08x,");
    }

    std::string dec() const {
        return elements_to_string("%10d,");
    }

    simd8uint32_tpl operator+(simd8uint32_tpl other) const {
        simd8uint32_tpl result;
        using namespace ppc64_vsx;
        st(result.u32, vec_add(ld(u32), ld(other.u32)));
        st(result.u32 + 4, vec_add(ld(u32 + 4), ld(other.u32 + 4)));
        return result;
    }

    simd8uint32_tpl operator-(simd8uint32_tpl other) const {
        simd8uint32_tpl result;
        using namespace ppc64_vsx;
        st(result.u32, vec_sub(ld(u32), ld(other.u32)));
        st(result.u32 + 4, vec_sub(ld(u32 + 4), ld(other.u32 + 4)));
        return result;
    }

    simd8uint32_tpl& operator+=(const simd8uint32_tpl& other) {
        using namespace ppc64_vsx;
        st(u32, vec_add(ld(u32), ld(other.u32)));
        st(u32 + 4, vec_add(ld(u32 + 4), ld(other.u32 + 4)));
        return *this;
    }

    bool operator==(simd8uint32_tpl other) const {
        using namespace ppc64_vsx;
        return vec_all_eq(ld(u32), ld(other.u32)) &&
                vec_all_eq(ld(u32 + 4), ld(other.u32 + 4));
    }

    bool operator!=(simd8uint32_tpl other) const {
        return !(*this == other);
    }

    // get scalar at index 0
    uint32_t get_scalar_0() const {
        return u32[0];
    }

    // for debugging only
    uint32_t operator[](int i) const {
        return u32[i];
    }

    void set1(uint32_t x) {
        using namespace ppc64_vsx;
        const ppc_u32x4 v = vec_splats(x);
        st(u32, v);
        st(u32 + 4, v);
    }

    simd8uint32_tpl unzip() const {
        using namespace ppc64_vsx;
        // (u0 u2 u4 u6 | u1 u3 u5 u7): even words of the 256-bit register,
        // then odd words
        const ppc_u32x4 h0 = ld(u32);
        const ppc_u32x4 h1 = ld(u32 + 4);
        simd8uint32_tpl ret;
        st(ret.u32, vec_perm(h0, h1, perm_even32()));
        st(ret.u32 + 4, vec_perm(h0, h1, perm_odd32()));
        return ret;
    }
};

inline void cmplt_min_max_fast(
        const simd8uint32_tpl<SIMDLevel::VSX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::VSX> candidateIndices,
        const simd8uint32_tpl<SIMDLevel::VSX> currentValues,
        const simd8uint32_tpl<SIMDLevel::VSX> currentIndices,
        simd8uint32_tpl<SIMDLevel::VSX>& minValues,
        simd8uint32_tpl<SIMDLevel::VSX>& minIndices,
        simd8uint32_tpl<SIMDLevel::VSX>& maxValues,
        simd8uint32_tpl<SIMDLevel::VSX>& maxIndices) {
    using namespace ppc64_vsx;
    for (int lane = 0; lane < 2; lane++) {
        const int o = 4 * lane;
        const ppc_u32x4 cv = ld(candidateValues.u32 + o);
        const ppc_u32x4 ci = ld(candidateIndices.u32 + o);
        const ppc_u32x4 uv = ld(currentValues.u32 + o);
        const ppc_u32x4 ui = ld(currentIndices.u32 + o);
        const ppc_u32x4 flag = (ppc_u32x4)vec_cmplt(cv, uv);
        st(minValues.u32 + o, vec_sel(uv, cv, flag));
        st(minIndices.u32 + o, vec_sel(ui, ci, flag));
        st(maxValues.u32 + o, vec_sel(cv, uv, flag));
        st(maxIndices.u32 + o, vec_sel(ci, ui, flag));
    }
}

template <>
struct simd8float32_tpl<SIMDLevel::VSX>
        : simd256bit_tpl<SIMDLevel::VSX> {
    simd8float32_tpl() {}

    explicit simd8float32_tpl(float x) {
        set1(x);
    }

    void set1(float x) {
        using namespace ppc64_vsx;
        const ppc_f32x4 v = vec_splats(x);
        st(f32, v);
        st(f32 + 4, v);
    }

    explicit simd8float32_tpl(
            float u0,
            float u1,
            float u2,
            float u3,
            float u4,
            float u5,
            float u6,
            float u7) {
        this->f32[0] = u0;
        this->f32[1] = u1;
        this->f32[2] = u2;
        this->f32[3] = u3;
        this->f32[4] = u4;
        this->f32[5] = u5;
        this->f32[6] = u6;
        this->f32[7] = u7;
    }

    explicit simd8float32_tpl(const simd256bit_tpl<SIMDLevel::VSX>& x)
            : simd256bit_tpl<SIMDLevel::VSX>(x) {}

    explicit simd8float32_tpl(const float* x)
            : simd256bit_tpl<SIMDLevel::VSX>((const void*)x) {}

    simd8float32_tpl operator*(const simd8float32_tpl& other) const {
        using namespace ppc64_vsx;
        simd8float32_tpl c;
        st(c.f32, vec_mul(ld(f32), ld(other.f32)));
        st(c.f32 + 4, vec_mul(ld(f32 + 4), ld(other.f32 + 4)));
        return c;
    }

    simd8float32_tpl operator+(const simd8float32_tpl& other) const {
        using namespace ppc64_vsx;
        simd8float32_tpl c;
        st(c.f32, vec_add(ld(f32), ld(other.f32)));
        st(c.f32 + 4, vec_add(ld(f32 + 4), ld(other.f32 + 4)));
        return c;
    }

    simd8float32_tpl operator-(const simd8float32_tpl& other) const {
        using namespace ppc64_vsx;
        simd8float32_tpl c;
        st(c.f32, vec_sub(ld(f32), ld(other.f32)));
        st(c.f32 + 4, vec_sub(ld(f32 + 4), ld(other.f32 + 4)));
        return c;
    }

    simd8float32_tpl& operator+=(const simd8float32_tpl& other) {
        using namespace ppc64_vsx;
        st(f32, vec_add(ld(f32), ld(other.f32)));
        st(f32 + 4, vec_add(ld(f32 + 4), ld(other.f32 + 4)));
        return *this;
    }

    bool operator==(simd8float32_tpl other) const {
        using namespace ppc64_vsx;
        return vec_all_eq(ld(f32), ld(other.f32)) &&
                vec_all_eq(ld(f32 + 4), ld(other.f32 + 4));
    }

    bool operator!=(simd8float32_tpl other) const {
        return !(*this == other);
    }

    std::string tostring() const {
        char res[1000], *ptr = res;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
        for (int i = 0; i < 8; i++) {
            ptr += snprintf(
                    ptr, (size_t)(res + sizeof(res) - ptr), "%g,", f32[i]);
        }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        // strip last ,
        ptr[-1] = 0;
        return std::string(res);
    }
};

// hadd does not cross lanes
inline simd8float32_tpl<SIMDLevel::VSX> hadd(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b) {
    simd8float32_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    const ppc_u8x16 pe = perm_even32();
    const ppc_u8x16 po = perm_odd32();
    for (int lane = 0; lane < 2; lane++) {
        const ppc_f32x4 al = ld(a.f32 + 4 * lane);
        const ppc_f32x4 bl = ld(b.f32 + 4 * lane);
        st(c.f32 + 4 * lane,
           vec_add(vec_perm(al, bl, pe), vec_perm(al, bl, po)));
    }
    return c;
}

inline simd8float32_tpl<SIMDLevel::VSX> unpacklo(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b) {
    simd8float32_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    st(c.f32, vec_mergeh(ld(a.f32), ld(b.f32)));
    st(c.f32 + 4, vec_mergeh(ld(a.f32 + 4), ld(b.f32 + 4)));
    return c;
}

inline simd8float32_tpl<SIMDLevel::VSX> unpackhi(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b) {
    simd8float32_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    st(c.f32, vec_mergel(ld(a.f32), ld(b.f32)));
    st(c.f32 + 4, vec_mergel(ld(a.f32 + 4), ld(b.f32 + 4)));
    return c;
}

// compute a * b + c
inline simd8float32_tpl<SIMDLevel::VSX> fmadd(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b,
        const simd8float32_tpl<SIMDLevel::VSX>& c) {
    simd8float32_tpl<SIMDLevel::VSX> res;
    using namespace ppc64_vsx;
    st(res.f32, vec_madd(ld(a.f32), ld(b.f32), ld(c.f32)));
    st(res.f32 + 4, vec_madd(ld(a.f32 + 4), ld(b.f32 + 4), ld(c.f32 + 4)));
    return res;
}


// get even float32's of a and b, interleaved
inline simd8float32_tpl<SIMDLevel::VSX> geteven(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b) {
    simd8float32_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    const ppc_u8x16 pe = perm_even32();
    st(c.f32, vec_perm(ld(a.f32), ld(b.f32), pe));
    st(c.f32 + 4, vec_perm(ld(a.f32 + 4), ld(b.f32 + 4), pe));
    return c;
}

// get odd float32's of a and b, interleaved
inline simd8float32_tpl<SIMDLevel::VSX> getodd(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b) {
    simd8float32_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    const ppc_u8x16 po = perm_odd32();
    st(c.f32, vec_perm(ld(a.f32), ld(b.f32), po));
    st(c.f32 + 4, vec_perm(ld(a.f32 + 4), ld(b.f32 + 4), po));
    return c;
}

// 3 cycles
// if the lanes are a = [a0 a1] and b = [b0 b1], return [a0 b0]
inline simd8float32_tpl<SIMDLevel::VSX> getlow128(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b) {
    simd8float32_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    st(c.f32, ld(a.f32));
    st(c.f32 + 4, ld(b.f32));
    return c;
}

inline simd8float32_tpl<SIMDLevel::VSX> gethigh128(
        const simd8float32_tpl<SIMDLevel::VSX>& a,
        const simd8float32_tpl<SIMDLevel::VSX>& b) {
    simd8float32_tpl<SIMDLevel::VSX> c;
    using namespace ppc64_vsx;
    st(c.f32, ld(a.f32 + 4));
    st(c.f32 + 4, ld(b.f32 + 4));
    return c;
}

// The following primitive is a vectorized version of the following code
// snippet:
//   float lowestValue = HUGE_VAL;
//   uint lowestIndex = 0;
//   for (size_t i = 0; i < n; i++) {
//     if (values[i] < lowestValue) {
//       lowestValue = values[i];
//       lowestIndex = i;
//     }
//   }
// Vectorized version can be implemented via two operations: cmp and blend
// with something like this:
//   lowestValues = [HUGE_VAL; 8];
//   lowestIndices = {0, 1, 2, 3, 4, 5, 6, 7};
//   for (size_t i = 0; i < n; i += 8) {
//     auto comparison = cmp(values + i, lowestValues);
//     lowestValues = blend(
//         comparison,
//         values + i,
//         lowestValues);
//     lowestIndices = blend(
//         comparison,
//         i + {0, 1, 2, 3, 4, 5, 6, 7},
//         lowestIndices);
//     lowestIndices += {8, 8, 8, 8, 8, 8, 8, 8};
//   }
// The problem is that blend primitive needs very different instruction
// order for AVX and ARM.
// So, let's introduce a combination of these two in order to avoid
// confusion for ppl who write in low-level SIMD instructions. Additionally,
// these two ops (cmp and blend) are very often used together.
inline void cmplt_and_blend_inplace(
        const simd8float32_tpl<SIMDLevel::VSX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::VSX> candidateIndices,
        simd8float32_tpl<SIMDLevel::VSX>& lowestValues,
        simd8uint32_tpl<SIMDLevel::VSX>& lowestIndices) {
    using namespace ppc64_vsx;
    for (int lane = 0; lane < 2; lane++) {
        const int o = 4 * lane;
        const ppc_f32x4 cv = ld(candidateValues.f32 + o);
        const ppc_f32x4 lv = ld(lowestValues.f32 + o);
        const ppc_u32x4 mask = (ppc_u32x4)vec_cmplt(cv, lv);
        st(lowestValues.f32 + o, vec_sel(lv, cv, mask));
        st(lowestIndices.u32 + o,
           vec_sel(ld(lowestIndices.u32 + o),
                   ld(candidateIndices.u32 + o),
                   mask));
    }
}

// Vectorized version of the following code:
//   for (size_t i = 0; i < n; i++) {
//      bool flag = (candidateValues[i] < currentValues[i]);
//      minValues[i] = flag ? candidateValues[i] : currentValues[i];
//      minIndices[i] = flag ? candidateIndices[i] : currentIndices[i];
//      maxValues[i] = !flag ? candidateValues[i] : currentValues[i];
//      maxIndices[i] = !flag ? candidateIndices[i] : currentIndices[i];
//   }
// Max indices evaluation is inaccurate in case of equal values (the index of
// the last equal value is saved instead of the first one), but this behavior
// saves instructions.
inline void cmplt_min_max_fast(
        const simd8float32_tpl<SIMDLevel::VSX> candidateValues,
        const simd8uint32_tpl<SIMDLevel::VSX> candidateIndices,
        const simd8float32_tpl<SIMDLevel::VSX> currentValues,
        const simd8uint32_tpl<SIMDLevel::VSX> currentIndices,
        simd8float32_tpl<SIMDLevel::VSX>& minValues,
        simd8uint32_tpl<SIMDLevel::VSX>& minIndices,
        simd8float32_tpl<SIMDLevel::VSX>& maxValues,
        simd8uint32_tpl<SIMDLevel::VSX>& maxIndices) {
    using namespace ppc64_vsx;
    for (int lane = 0; lane < 2; lane++) {
        const int o = 4 * lane;
        const ppc_f32x4 cv = ld(candidateValues.f32 + o);
        const ppc_u32x4 ci = ld(candidateIndices.u32 + o);
        const ppc_f32x4 uv = ld(currentValues.f32 + o);
        const ppc_u32x4 ui = ld(currentIndices.u32 + o);
        const ppc_u32x4 flag = (ppc_u32x4)vec_cmplt(cv, uv);
        st(minValues.f32 + o, vec_sel(uv, cv, flag));
        st(minIndices.u32 + o, vec_sel(ui, ci, flag));
        st(maxValues.f32 + o, vec_sel(cv, uv, flag));
        st(maxIndices.u32 + o, vec_sel(ci, ui, flag));
    }
}

// horizontal add: sum all 8 floats in the register
inline float horizontal_add(const simd8float32_tpl<SIMDLevel::VSX>& a) {
    using namespace ppc64_vsx;
    const ppc_f32x4 s = vec_add(ld(a.f32), ld(a.f32 + 4));
    float t[4];
    st(t, s);
    return (t[0] + t[1]) + (t[2] + t[3]);
}

#endif // FAISS_SIMDLIB_PPC64_VSX

} // namespace faiss
