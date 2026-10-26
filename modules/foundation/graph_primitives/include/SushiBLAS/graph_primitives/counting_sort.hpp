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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/primitive_options.hpp>

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
         * @brief Records a stable argsort of uint32 keys by radix digits, with bucket offsets.
         * @param indices Receives the sorted order, as indices into the keys window.
         * @param offsets Receives the bucket starts; holds key_range.capacity() + 1 elements.
         * @param counts  Optional bucket sizes; holds key_range.capacity() elements.
         * @param options Names and classes every node; none leaves them unnamed and Unstated.
         * @return Nothing, or an error; modules/foundation/graph_primitives/README.md lists them.
         */
        [[nodiscard]] SushiRuntime::Result<void>
        counting_sort(SushiRuntime::Recorder& recorder,
                      const SushiRuntime::Buffer<std::uint32_t>& keys,
                      const SushiRuntime::Buffer<std::uint32_t>& indices,
                      const SushiRuntime::Buffer<std::uint32_t>& offsets,
                      const SushiRuntime::Buffer<std::uint32_t>& total,
                      SushiRuntime::LateCount n,
                      SushiRuntime::LateCount key_range,
                      const SushiRuntime::Buffer<std::uint32_t>* counts = nullptr,
                      CountingSortWindow window = {},
                      const std::optional<PrimitiveOptions>& options = std::nullopt);

    } // namespace GraphPrimitives
} // namespace SushiBLAS
