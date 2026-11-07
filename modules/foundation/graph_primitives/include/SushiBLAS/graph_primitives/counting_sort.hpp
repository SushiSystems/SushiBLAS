/**************************************************************************/
/* counting_sort.hpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file counting_sort.hpp
 * @brief Declares counting_sort(), a stable argsort of uint32 keys into buckets.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include <SushiBLAS/graph_primitives/runtime_primitive_recorder.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Places the operands of counting_sort() inside their buffers, in elements.
         * @see modules/foundation/graph_primitives/README.md
         */
        struct CountingSortWindow
        {
            std::size_t keys_offset = 0;    /**< Element of keys that element 0 names. */
            std::size_t indices_offset = 0; /**< Element of indices the order starts at. */
            std::size_t offsets_offset = 0; /**< Element of offsets bucket 0 starts at. */
            std::size_t total_offset = 0;   /**< Element of total that receives the count. */
            std::size_t counts_offset = 0;  /**< Element of counts bucket 0 sizes. */
        };

        /**
         * @brief Records a stable argsort of uint32 keys with bucket offsets, on either recorder.
         * @param indices Receives the sorted order, as indices into the keys window.
         * @param offsets Receives the bucket starts; holds key_range.capacity() + 1 elements.
         * @param counts  Optional bucket sizes; holds key_range.capacity() elements.
         * @param options Names and classes every node; none leaves them unnamed and Unstated.
         * @return Nothing, or an error; modules/foundation/graph_primitives/README.md lists them.
         */
        template <class Port>
        [[nodiscard]] SushiRuntime::Result<void>
        counting_sort(Port& recorder,
                      const typename Port::template Buffer<std::uint32_t>& keys,
                      const typename Port::template Buffer<std::uint32_t>& indices,
                      const typename Port::template Buffer<std::uint32_t>& offsets,
                      const typename Port::template Buffer<std::uint32_t>& total,
                      SushiRuntime::LateCount n,
                      SushiRuntime::LateCount key_range,
                      const typename Port::template Buffer<std::uint32_t>* counts = nullptr,
                      CountingSortWindow window = {},
                      const std::optional<PrimitiveOptions>& options = std::nullopt);

    } // namespace GraphPrimitives
} // namespace SushiBLAS
