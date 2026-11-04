/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_VSX_H
#define HAMMING_COMPUTER_VSX_H

// PowerPC VSX-optimized HammingComputer and GenHammingComputer specializations.
//
// On IBM POWER (POWER7+), the scalar integer unit contains a dedicated
// hardware instruction `popcntd` (Population Count Doubleword) with 2-cycle
// latency and 2-per-cycle throughput in General Purpose Registers (GPR).
//
// For standard Hamming distance (counting differing bits):
// - Using VSX byte-popcount (`vpopcntb`) requires horizontal tree reduction
//   (`vsum4s`, `vsldoi`, `vadduwm`) and a high-latency cross-domain vector-to-GPR
//   transfer (`mfvsrwz`).
// - Therefore, following the AVX2 strategy (which similarly avoids SIMD for
//   standard Hamming because `_mm_popcnt_u64` is faster), the standard
//   HammingComputers on POWER utilize superscalar pipelined 64-bit `popcntd`
//   with balanced accumulator trees to achieve maximum throughput.
//
// For Generalized Hamming distance (counting differing bytes):
// - VSX vector byte equality (`vec_cmpeq`) operates on 16 bytes at once in 1
//   cycle, which is significantly faster than scalar bit-twiddling, and is
//   fully vectorized using VSX instructions.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/hamming_distance/common.h>
#include <faiss/utils/hamming_distance/hamming_computer-generic.h>
#include <faiss/utils/popcount.h>

#if defined(__PPC64__) && defined(__VSX__)
#include <altivec.h>
#undef vector
#undef pixel
#undef bool

namespace faiss {

/******************************************************************
 * Standard HammingComputer specializations for PowerPC.
 * Optimized with superscalar popcntd and balanced accumulator trees.
 ******************************************************************/

template <>
struct HammingComputer16_tpl<SIMDLevel::VSX> {
    uint64_t a0, a1;

    HammingComputer16_tpl() {}

    HammingComputer16_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct HammingComputer20_tpl<SIMDLevel::VSX> {
    uint64_t a0, a1;
    uint32_t a2;

    HammingComputer20_tpl() {}

    HammingComputer20_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 20);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        a2 = a32[4];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b8);
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
                popcount32(b32[4] ^ a2);
    }

    inline static constexpr int get_code_size() {
        return 20;
    }
};

template <>
struct HammingComputer32_tpl<SIMDLevel::VSX> {
    uint64_t a0, a1, a2, a3;

    HammingComputer32_tpl() {}

    HammingComputer32_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        // Independent accumulator trees allow dual-issue popcntd execution
        int s0 = popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1);
        int s1 = popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3);
        return s0 + s1;
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct HammingComputer64_tpl<SIMDLevel::VSX> {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;

    HammingComputer64_tpl() {}

    HammingComputer64_tpl(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 64);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
        a4 = a[4];
        a5 = a[5];
        a6 = a[6];
        a7 = a[7];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        int s0 = popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1);
        int s1 = popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3);
        int s2 = popcount64(b[4] ^ a4) + popcount64(b[5] ^ a5);
        int s3 = popcount64(b[6] ^ a6) + popcount64(b[7] ^ a7);
        return (s0 + s1) + (s2 + s3);
    }

    inline static constexpr int get_code_size() {
        return 64;
    }
};

template <>
struct HammingComputerDefault_tpl<SIMDLevel::VSX> {
    const uint8_t* a8;
    int quotient8;
    int remainder8;

    HammingComputerDefault_tpl() {}

    HammingComputerDefault_tpl(const uint8_t* a8_in, int code_size) {
        set(a8_in, code_size);
    }

    void set(const uint8_t* a8_in, int code_size) {
        this->a8 = a8_in;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        int i = 0;
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;

        // Process 8 doublewords (64 bytes) per iteration with 4 parallel accumulators
        for (; i + 8 <= quotient8; i += 8) {
            s0 += popcount64(a64[i] ^ b64[i]);
            s1 += popcount64(a64[i + 1] ^ b64[i + 1]);
            s2 += popcount64(a64[i + 2] ^ b64[i + 2]);
            s3 += popcount64(a64[i + 3] ^ b64[i + 3]);
            s0 += popcount64(a64[i + 4] ^ b64[i + 4]);
            s1 += popcount64(a64[i + 5] ^ b64[i + 5]);
            s2 += popcount64(a64[i + 6] ^ b64[i + 6]);
            s3 += popcount64(a64[i + 7] ^ b64[i + 7]);
        }

        // Process remaining full doublewords
        for (; i < quotient8; i++) {
            s0 += popcount64(a64[i] ^ b64[i]);
        }

        int total = (s0 + s1) + (s2 + s3);

        if (remainder8) {
            const uint8_t* a = a8 + 8 * quotient8;
            const uint8_t* b = b8 + 8 * quotient8;
            int rem = remainder8;
            if (rem >= 4) {
                uint32_t a32, b32;
                memcpy(&a32, a, 4);
                memcpy(&b32, b, 4);
                total += popcount32(a32 ^ b32);
                a += 4;
                b += 4;
                rem -= 4;
            }
            if (rem >= 2) {
                uint16_t a16, b16;
                memcpy(&a16, a, 2);
                memcpy(&b16, b, 2);
                total += popcount32(a16 ^ b16);
                a += 2;
                b += 2;
                rem -= 2;
            }
            if (rem >= 1) {
                total += popcount32(*a ^ *b);
            }
        }
        return total;
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};

/***************************************************************************
 * VSX-vectorized Generalized Hamming Computer specializations.
 * Vector byte equality (`vec_cmpeq`) is significantly faster than scalar
 * bit-twiddling (`generalized_hamming_64`).
 ***************************************************************************/

namespace vsx {

using ppc_u8x16 = __vector unsigned char;
using ppc_u32x4 = __vector unsigned int;

inline ppc_u8x16 ld(const uint8_t* p) {
    return vec_xl(0, p);
}

// Horizontally sum four 32-bit words in a vector across all lanes
inline int vsum_u32(ppc_u32x4 s4) {
    const ppc_u32x4 s2 = vec_add(s4, vec_sld(s4, s4, 8));
    const ppc_u32x4 s1 = vec_add(s2, vec_sld(s2, s2, 4));
#if defined(__GNUC__) || defined(__clang__)
    return vec_extract(s1, 0);
#else
    return s1[0];
#endif
}

// Horizontally sum all 16 bytes in a vector
inline int vsum_u8(ppc_u8x16 cnt) {
    const ppc_u32x4 zero = vec_splats(0u);
    const ppc_u32x4 s4 = vec_sum4s(cnt, zero);
    return vsum_u32(s4);
}

} // namespace vsx

template <>
struct GenHammingComputer8_tpl<SIMDLevel::VSX>
        : GenHammingComputer8_tpl<SIMDLevel::NONE> {
    using GenHammingComputer8_tpl<SIMDLevel::NONE>::GenHammingComputer8_tpl;
};

template <>
struct GenHammingComputer16_tpl<SIMDLevel::VSX> {
    vsx::ppc_u8x16 a0;

    GenHammingComputer16_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        a0 = vsx::ld(a8);
    }

    inline int hamming(const uint8_t* b8) const {
        vsx::ppc_u8x16 b0 = vsx::ld(b8);
        vsx::ppc_u8x16 eq = (vsx::ppc_u8x16)vec_cmpeq(a0, b0);
        vsx::ppc_u8x16 diff = vec_andc(vec_splats((uint8_t)1), eq);
        return vsx::vsum_u8(diff);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct GenHammingComputer32_tpl<SIMDLevel::VSX> {
    vsx::ppc_u8x16 a0, a1;

    GenHammingComputer32_tpl(
            const uint8_t* a8,
            FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        a0 = vsx::ld(a8);
        a1 = vsx::ld(a8 + 16);
    }

    inline int hamming(const uint8_t* b8) const {
        vsx::ppc_u8x16 b0 = vsx::ld(b8);
        vsx::ppc_u8x16 b1 = vsx::ld(b8 + 16);
        vsx::ppc_u8x16 eq0 = (vsx::ppc_u8x16)vec_cmpeq(a0, b0);
        vsx::ppc_u8x16 eq1 = (vsx::ppc_u8x16)vec_cmpeq(a1, b1);
        const vsx::ppc_u8x16 one = vec_splats((uint8_t)1);
        vsx::ppc_u8x16 diff0 = vec_andc(one, eq0);
        vsx::ppc_u8x16 diff1 = vec_andc(one, eq1);
        return vsx::vsum_u8(vec_add(diff0, diff1));
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct GenHammingComputerM8_tpl<SIMDLevel::VSX> {
    const uint8_t* a;
    int n;

    GenHammingComputerM8_tpl(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = a8;
        n = code_size;
    }

    int hamming(const uint8_t* b) const {
        const vsx::ppc_u32x4 zero = vec_splats(0u);
        const vsx::ppc_u8x16 one = vec_splats((uint8_t)1);
        vsx::ppc_u32x4 acc32 = zero;
        vsx::ppc_u8x16 acc8 = vec_splats((uint8_t)0);

        int i = 0;
        int n_acc8 = 0;
        for (; i + 32 <= n; i += 32) {
            vsx::ppc_u8x16 a0 = vsx::ld(a + i);
            vsx::ppc_u8x16 b0 = vsx::ld(b + i);
            vsx::ppc_u8x16 a1 = vsx::ld(a + i + 16);
            vsx::ppc_u8x16 b1 = vsx::ld(b + i + 16);
            vsx::ppc_u8x16 eq0 = (vsx::ppc_u8x16)vec_cmpeq(a0, b0);
            vsx::ppc_u8x16 eq1 = (vsx::ppc_u8x16)vec_cmpeq(a1, b1);
            vsx::ppc_u8x16 diff0 = vec_andc(one, eq0);
            vsx::ppc_u8x16 diff1 = vec_andc(one, eq1);
            acc8 = vec_add(acc8, vec_add(diff0, diff1));
            n_acc8 += 2;
            if (n_acc8 >= 16) {
                acc32 = vec_add(acc32, vec_sum4s(acc8, zero));
                acc8 = vec_splats((uint8_t)0);
                n_acc8 = 0;
            }
        }
        for (; i + 16 <= n; i += 16) {
            vsx::ppc_u8x16 a0 = vsx::ld(a + i);
            vsx::ppc_u8x16 b0 = vsx::ld(b + i);
            vsx::ppc_u8x16 eq0 = (vsx::ppc_u8x16)vec_cmpeq(a0, b0);
            acc8 = vec_add(acc8, vec_andc(one, eq0));
            n_acc8++;
        }
        if (n_acc8 > 0) {
            acc32 = vec_add(acc32, vec_sum4s(acc8, zero));
        }
        int total = vsx::vsum_u32(acc32);
        if (i < n) {
            uint64_t a64, b64;
            memcpy(&a64, a + i, 8);
            memcpy(&b64, b + i, 8);
            total += generalized_hamming_64(a64 ^ b64);
        }
        return total;
    }

    inline int get_code_size() const {
        return n;
    }
};

} // namespace faiss

#else // !defined(__PPC64__) || !defined(__VSX__)

namespace faiss {

#define FAISS_INHERIT_HAMMING_VSX(Class)                                      \
    template <>                                                               \
    struct Class##_tpl<SIMDLevel::VSX> : Class##_tpl<SIMDLevel::NONE> {       \
        using Class##_tpl<SIMDLevel::NONE>::Class##_tpl;                      \
    }

FAISS_INHERIT_HAMMING_VSX(HammingComputer16);
FAISS_INHERIT_HAMMING_VSX(HammingComputer20);
FAISS_INHERIT_HAMMING_VSX(HammingComputer32);
FAISS_INHERIT_HAMMING_VSX(HammingComputer64);
FAISS_INHERIT_HAMMING_VSX(HammingComputerDefault);
FAISS_INHERIT_HAMMING_VSX(GenHammingComputer8);
FAISS_INHERIT_HAMMING_VSX(GenHammingComputer16);
FAISS_INHERIT_HAMMING_VSX(GenHammingComputer32);
FAISS_INHERIT_HAMMING_VSX(GenHammingComputerM8);

#undef FAISS_INHERIT_HAMMING_VSX

} // namespace faiss

#endif // defined(__PPC64__) && defined(__VSX__)

#endif // HAMMING_COMPUTER_VSX_H
