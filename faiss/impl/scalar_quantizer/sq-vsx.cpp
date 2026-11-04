/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>

namespace faiss {
namespace scalar_quantizer {

template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<SIMDLevel::VSX>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_quantizer<SIMDLevel::NONE>(qtype, d, trained);
}

template <>
SQDistanceComputer* sq_select_distance_computer<SIMDLevel::VSX>(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_distance_computer<SIMDLevel::NONE>(
            metric, qtype, d, trained);
}

template <>
InvertedListScanner* sq_select_InvertedListScanner<SIMDLevel::VSX>(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    return sq_select_InvertedListScanner<SIMDLevel::NONE>(
            qtype,
            mt,
            d,
            code_size,
            trained,
            quantizer,
            store_pairs,
            sel,
            by_residual);
}

} // namespace scalar_quantizer
} // namespace faiss
