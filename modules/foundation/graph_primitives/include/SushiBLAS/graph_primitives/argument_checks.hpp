/**************************************************************************/
/* argument_checks.hpp                                                    */
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

/**
 * @file argument_checks.hpp
 * @brief Declares the value-level argument checks the recording primitives run first.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <optional>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/element_window.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Names one operand by the facts a check needs: its buffer, window and device.
             * @see modules/foundation/graph_primitives/README.md
             */
            struct Operand
            {
                std::size_t size = 0;             /**< Elements the buffer holds. */
                std::size_t offset = 0;           /**< First element of the window. */
                SushiRuntime::DeviceIndex device; /**< Device the buffer lives on. */
            };

            /**
             * @brief Holds the facts counting_sort() checks before it records anything.
             * @see modules/foundation/graph_primitives/README.md
             */
            struct CountingSortShape
            {
                std::size_t n_capacity = 0;    /**< Capacity of the element count. */
                std::size_t key_capacity = 0;  /**< Capacity of the key range. */
                Operand keys;                  /**< The keys buffer and its window. */
                Operand indices;               /**< The indices buffer and its window. */
                Operand offsets;               /**< The offsets buffer and its window. */
                Operand total;                 /**< The total buffer and its window. */
                std::optional<Operand> counts; /**< The counts buffer; none when not asked for. */
            };

            /**
             * @brief Holds the facts prefix_scan() checks before it records anything.
             * @see modules/foundation/graph_primitives/README.md
             */
            struct PrefixScanShape
            {
                std::size_t capacity = 0; /**< Capacity of the element count. */
                Operand counts;           /**< The counts buffer and its window. */
                Operand offsets;          /**< The offsets buffer and its window. */
                Operand total;            /**< The total buffer and its window. */
            };

            /**
             * @brief Holds the facts reduce() checks before it records anything.
             * @see modules/foundation/graph_primitives/README.md
             */
            struct ReduceShape
            {
                std::size_t count = 0; /**< Elements folded, or the capacity of a late count. */
                Operand input;         /**< The input buffer and its window. */
                Operand output;        /**< The output buffer and its window. */
            };

            /**
             * @brief Holds the facts segmented_reduce() checks before it records anything.
             * @see modules/foundation/graph_primitives/README.md
             */
            struct SegmentedReduceShape
            {
                std::size_t segment_count = 0; /**< Segments folded, or the capacity. */
                Operand values;                /**< The values buffer and its window. */
                std::size_t values_count = 0;  /**< Elements of the values window. */
                Operand segment_offsets;       /**< The segment offsets buffer and its window. */
                Operand output;                /**< The output buffer and its window. */
            };

            /**
             * @brief Checks the options and every operand of a counting_sort() call.
             * @return Nothing, or the InvalidGraph error of the first rule broken.
             */
            [[nodiscard]] inline SushiRuntime::Result<void>
            validate_counting_sort(const CountingSortShape& shape,
                                   const std::optional<PrimitiveOptions>& options)
            {
                SushiRuntime::Result<void> named = require_options("counting_sort", options);
                if (!named)
                    return named;
                if (shape.key_capacity == 0)
                    return invalid_graph("counting_sort: key_range.capacity() is zero.");

                SushiRuntime::Result<void> fits = require_window(
                    "counting_sort", "keys", shape.keys.size, shape.keys.offset, shape.n_capacity);
                if (fits)
                    fits = require_window("counting_sort", "indices", shape.indices.size,
                                          shape.indices.offset, shape.n_capacity);
                if (fits)
                    fits = require_window("counting_sort", "offsets", shape.offsets.size,
                                          shape.offsets.offset, shape.key_capacity + 1);
                if (fits)
                    fits = require_window("counting_sort", "total", shape.total.size,
                                          shape.total.offset, std::size_t{1});
                if (fits && shape.counts)
                    fits = require_window("counting_sort", "counts", shape.counts->size,
                                          shape.counts->offset, shape.key_capacity);
                if (!fits)
                    return fits;

                if (shape.indices.device != shape.keys.device ||
                    shape.offsets.device != shape.keys.device ||
                    (shape.counts && shape.counts->device != shape.keys.device))
                    return invalid_graph("counting_sort: keys, indices, offsets and counts must "
                                         "live on the same device.");
                return {};
            }

            /**
             * @brief Checks the options and every operand of a prefix_scan() call.
             * @return Nothing, or the InvalidGraph error of the first rule broken.
             */
            [[nodiscard]] inline SushiRuntime::Result<void>
            validate_prefix_scan(const PrefixScanShape& shape,
                                 const std::optional<PrimitiveOptions>& options)
            {
                SushiRuntime::Result<void> named = require_options("prefix_scan", options);
                if (!named)
                    return named;

                SushiRuntime::Result<void> fits = require_window(
                    "prefix_scan", "counts", shape.counts.size, shape.counts.offset,
                    shape.capacity);
                if (fits)
                    fits = require_window("prefix_scan", "offsets", shape.offsets.size,
                                          shape.offsets.offset, shape.capacity + 1);
                if (fits)
                    fits = require_window("prefix_scan", "total", shape.total.size,
                                          shape.total.offset, std::size_t{1});
                if (!fits)
                    return fits;

                if (shape.offsets.device != shape.counts.device)
                    return invalid_graph(
                        "prefix_scan: counts and offsets must live on the same device.");
                return {};
            }

            /**
             * @brief Checks the options and both operands of a reduce() call.
             * @return Nothing, or the InvalidGraph error of the first rule broken.
             */
            [[nodiscard]] inline SushiRuntime::Result<void>
            validate_reduce(const ReduceShape& shape,
                            const std::optional<PrimitiveOptions>& options)
            {
                SushiRuntime::Result<void> named = require_options("reduce", options);
                if (!named)
                    return named;

                SushiRuntime::Result<void> fits = require_window(
                    "reduce", "input", shape.input.size, shape.input.offset, shape.count);
                if (fits)
                    fits = require_window("reduce", "output", shape.output.size,
                                          shape.output.offset, std::size_t{1});
                return fits;
            }

            /**
             * @brief Checks the options and every operand of a segmented_reduce() call.
             * @return Nothing, or the InvalidGraph error of the first rule broken.
             */
            [[nodiscard]] inline SushiRuntime::Result<void>
            validate_segmented_reduce(const SegmentedReduceShape& shape,
                                      const std::optional<PrimitiveOptions>& options)
            {
                SushiRuntime::Result<void> named = require_options("segmented_reduce", options);
                if (!named)
                    return named;

                SushiRuntime::Result<void> fits = require_window(
                    "segmented_reduce", "values", shape.values.size, shape.values.offset,
                    shape.values_count);
                if (fits)
                    fits = require_window("segmented_reduce", "segment_offsets",
                                          shape.segment_offsets.size, shape.segment_offsets.offset,
                                          shape.segment_count + 1);
                if (fits)
                    fits = require_window("segmented_reduce", "output", shape.output.size,
                                          shape.output.offset, shape.segment_count);
                return fits;
            }
        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
