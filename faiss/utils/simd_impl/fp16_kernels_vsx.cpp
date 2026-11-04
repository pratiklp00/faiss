/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_impl/fp16_kernels.h>

namespace faiss {
namespace detail {

template <>
void fp16_madd<SIMDLevel::VSX>(
        size_t d,
        const uint16_t* x,
        float w,
        float* c) {
    for (size_t j = 0; j < d; ++j) {
        c[j] += decode_fp16(x[j]) * w;
    }
}

template <>
void fp16_to_fp32_kernel<SIMDLevel::VSX>(
        size_t n,
        const uint16_t* x,
        float* out) {
    for (size_t j = 0; j < n; ++j) {
        out[j] = decode_fp16(x[j]);
    }
}

} // namespace detail
} // namespace faiss
