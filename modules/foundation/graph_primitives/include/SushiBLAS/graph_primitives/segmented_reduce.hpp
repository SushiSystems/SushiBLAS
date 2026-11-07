/**************************************************************************/
/* segmented_reduce.hpp                                                   */
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
 * @file segmented_reduce.hpp
 * @brief Declares segmented_reduce(), a fixed-order reduction of each segment of a ragged array.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/device_sized.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>

#include <SushiBLAS/graph_primitives/argument_checks.hpp>
#include <SushiBLAS/graph_primitives/element_window.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>
#include <SushiBLAS/graph_primitives/node_stamper.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include <SushiBLAS/graph_primitives/primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/segment_fold.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Places the operands of segmented_reduce() inside their buffers, in elements.
         * @see modules/foundation/graph_primitives/README.md
         */
        struct SegmentedReduceWindow
        {
            /** @brief Holds the values_count that runs the values window to its buffer's end. */
            static constexpr std::size_t K_TO_END = ~std::size_t{0};

            std::size_t values_offset = 0;      /**< Element of values that offset 0 names. */
            std::size_t values_count = K_TO_END; /**< Elements of values the segments index. */
            std::size_t offsets_offset = 0;     /**< Element of segment_offsets read first. */
            std::size_t output_offset = 0;      /**< Element of output segment 0 writes. */
        };

        namespace Detail
        {
            /**
             * @brief Checks a segmented reduction's call, then returns its three spans.
             * @return Nothing, or the InvalidGraph error of the first rule broken.
             */
            template <typename T>
            [[nodiscard]] SushiRuntime::Result<void>
            segment_spans(const SushiRuntime::Buffer<T>& values,
                          const SushiRuntime::Buffer<std::uint32_t>& segment_offsets,
                          const SushiRuntime::Buffer<T>& output, std::size_t segment_count,
                          const SegmentedReduceWindow& window,
                          const std::optional<PrimitiveOptions>& options,
                          ElementSpan<T>& value_span, ElementSpan<std::uint32_t>& offset_span,
                          ElementSpan<T>& output_span)
            {
                const std::size_t value_count =
                    (window.values_count == SegmentedReduceWindow::K_TO_END)
                        ? (window.values_offset <= values.size()
                               ? values.size() - window.values_offset
                               : 0)
                        : window.values_count;

                SegmentedReduceShape shape;
                shape.segment_count = segment_count;
                shape.values =
                    Operand{values.size(), window.values_offset, values.device_index()};
                shape.values_count = value_count;
                shape.segment_offsets = Operand{segment_offsets.size(), window.offsets_offset,
                                                segment_offsets.device_index()};
                shape.output =
                    Operand{output.size(), window.output_offset, output.device_index()};
                SushiRuntime::Result<void> checked = validate_segmented_reduce(shape, options);
                if (!checked)
                    return checked;

                value_span = elements(values, window.values_offset, value_count);
                offset_span = elements(segment_offsets, window.offsets_offset, segment_count + 1);
                output_span = elements(output, window.output_offset, segment_count);
                return {};
            }
        } // namespace Detail

        /**
         * @brief Records a reduction of each segment of a ragged array, in a fixed order.
         * @tparam Combine Binary functor; must be trivially copyable.
         * @param segment_offsets segment_count + 1 non-decreasing indices into the values window.
         * @param output Receives one result per segment; identity for an empty segment.
         * @param options Names and classes every node; none leaves them unnamed and Unstated.
         * @return Nothing, InvalidGraph (a window or options), or the recorder's refusal.
         */
        template <class Port, typename T, class Combine,
                  std::enable_if_t<K_IS_PRIMITIVE_RECORDER<Port>, int> = 0>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(Port& recorder,
                         const SushiRuntime::Buffer<T>& values,
                         const SushiRuntime::Buffer<std::uint32_t>& segment_offsets,
                         const SushiRuntime::Buffer<T>& output,
                         std::size_t segment_count, Combine op, T identity,
                         SegmentedReduceWindow window = {},
                         const std::optional<PrimitiveOptions>& options = std::nullopt)
        {
            using SushiRuntime::Access;
            using SushiRuntime::Reads;
            using SushiRuntime::Writes;

            Detail::NodeStamper stamper(options, "graph.segmented_reduce");

            Detail::ElementSpan<T> value_span;
            Detail::ElementSpan<std::uint32_t> offset_span;
            Detail::ElementSpan<T> output_span;
            SushiRuntime::Result<void> fits =
                Detail::segment_spans(values, segment_offsets, output, segment_count, window,
                                      options, value_span, offset_span, output_span);
            if (!fits)
                return fits;

            const T* value_base = value_span.data();
            const std::uint32_t* offset_base = offset_span.data();
            T* out_base = output_span.data();

            recorder.kernel(segment_count,
                Access(Detail::access_of<Reads>(value_span, offset_span),
                       Detail::access_of<Writes>(output_span)),
                [value_base, offset_base, out_base, op, identity](std::size_t s)
                {
                    out_base[s] = Detail::fold_segment(op, value_base, offset_base, s, identity);
                },
                stamper.node(segment_count));

            return Detail::recording_outcome(recorder);
        }

        /**
         * @brief Records a segmented reduction whose segment count is resolved once per run.
         * @tparam Combine Binary functor; must be trivially copyable.
         * @param segment_offsets segment_count.capacity() + 1 indices into the values window.
         * @param output segment_count.capacity() entries; those past the live count are kept.
         * @param options Names and classes every node; none leaves them unnamed and Unstated.
         * @return As the fixed-count overload, or AllocationFailed; a run fails past the capacity.
         */
        template <class Port, typename T, class Combine,
                  std::enable_if_t<K_IS_PRIMITIVE_RECORDER<Port>, int> = 0>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(Port& recorder,
                         const SushiRuntime::Buffer<T>& values,
                         const SushiRuntime::Buffer<std::uint32_t>& segment_offsets,
                         const SushiRuntime::Buffer<T>& output,
                         SushiRuntime::LateCount segment_count, Combine op, T identity,
                         SegmentedReduceWindow window = {},
                         const std::optional<PrimitiveOptions>& options = std::nullopt)
        {
            using SushiRuntime::Access;
            using SushiRuntime::DeviceSized;
            using SushiRuntime::LateCount;
            using SushiRuntime::Reads;
            using SushiRuntime::Writes;
            using SushiRuntime::BufferRegion;

            Detail::NodeStamper stamper(options, "graph.segmented_reduce");

            const std::size_t capacity = segment_count.capacity();
            Detail::ElementSpan<T> value_span;
            Detail::ElementSpan<std::uint32_t> offset_span;
            Detail::ElementSpan<T> output_span;
            SushiRuntime::Result<void> fits =
                Detail::segment_spans(values, segment_offsets, output, capacity, window, options,
                                      value_span, offset_span, output_span);
            if (!fits)
                return fits;

            const T* value_base = value_span.data();
            const std::uint32_t* offset_base = offset_span.data();
            T* out_base = output_span.data();

            // state[0] is the live segment count the launch reads on the device.
            auto state_block = recorder.template scratch<std::uint32_t>(
                1, values.device_index(), values.residency());
            if (!state_block)
                return SushiRuntime::make_error(std::move(state_block.error()));
            std::uint32_t* state = *state_block;
            const BufferRegion state_region{state, 0, sizeof(std::uint32_t)};

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            recorder.resolve(std::array<LateCount, 1>{segment_count}, values.device_index(),
                Access(Reads(), Writes(state_region)),
                [state](const std::array<std::size_t, 1>& live)
                {
                    state[0] = static_cast<std::uint32_t>(live[0]);
                },
                stamper.stage("resolve", std::size_t{1}));

            recorder.kernel(
                DeviceSized{capacity, state, nullptr},
                Access(Detail::access_of<Reads>(value_span, offset_span, state_region),
                       Detail::access_of<Writes>(output_span)),
                [value_base, offset_base, out_base, op, identity](std::size_t s)
                {
                    out_base[s] = Detail::fold_segment(op, value_base, offset_base, s, identity);
                },
                stamper.stage("segment", capacity));

            return Detail::recording_outcome(recorder);
        }

        /**
         * @brief Records a fixed-count segmented reduction with Combine::identity().
         * @tparam Combine Binary functor with a static identity(); must be trivially copyable.
         * @return As the overload that takes an identity.
         */
        template <class Port, typename T, class Combine,
                  std::enable_if_t<K_IS_PRIMITIVE_RECORDER<Port>, int> = 0>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(Port& recorder,
                         const SushiRuntime::Buffer<T>& values,
                         const SushiRuntime::Buffer<std::uint32_t>& segment_offsets,
                         const SushiRuntime::Buffer<T>& output,
                         std::size_t segment_count, Combine op,
                         SegmentedReduceWindow window = {},
                         const std::optional<PrimitiveOptions>& options = std::nullopt)
        {
            return GraphPrimitives::segmented_reduce(recorder, values, segment_offsets, output,
                                                     segment_count, op, Combine::identity(),
                                                     window, options);
        }

        /**
         * @brief Records a late-bound segmented reduction with Combine::identity().
         * @tparam Combine Binary functor with a static identity(); must be trivially copyable.
         * @return As the overload that takes an identity.
         */
        template <class Port, typename T, class Combine,
                  std::enable_if_t<K_IS_PRIMITIVE_RECORDER<Port>, int> = 0>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(Port& recorder,
                         const SushiRuntime::Buffer<T>& values,
                         const SushiRuntime::Buffer<std::uint32_t>& segment_offsets,
                         const SushiRuntime::Buffer<T>& output,
                         SushiRuntime::LateCount segment_count, Combine op,
                         SegmentedReduceWindow window = {},
                         const std::optional<PrimitiveOptions>& options = std::nullopt)
        {
            return GraphPrimitives::segmented_reduce(recorder, values, segment_offsets, output,
                                                     segment_count, op, Combine::identity(),
                                                     window, options);
        }

    } // namespace GraphPrimitives
} // namespace SushiBLAS
