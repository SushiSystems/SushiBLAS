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
 * @file segmented_reduce.hpp
 * @brief Declares segmented_reduce(), a fixed-order reduction of each segment of a ragged array.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/device_sized.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>

#include <SushiBLAS/graph_primitives/element_window.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>
#include <SushiBLAS/graph_primitives/node_stamper.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>

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
             * @brief Returns the spans of a segmented reduction, or why one does not fit.
             * @return Nothing, or an InvalidGraph error naming the operand whose window is short.
             */
            template <typename T>
            [[nodiscard]] SushiRuntime::Result<void>
            segment_spans(const SushiRuntime::Buffer<T>& values,
                          const SushiRuntime::Buffer<std::uint32_t>& segment_offsets,
                          const SushiRuntime::Buffer<T>& output, std::size_t segment_count,
                          const SegmentedReduceWindow& window, ElementSpan<T>& value_span,
                          ElementSpan<std::uint32_t>& offset_span, ElementSpan<T>& output_span)
            {
                const std::size_t value_count =
                    (window.values_count == SegmentedReduceWindow::K_TO_END)
                        ? (window.values_offset <= values.size()
                               ? values.size() - window.values_offset
                               : 0)
                        : window.values_count;
                value_span = elements(values, window.values_offset, value_count);
                offset_span = elements(segment_offsets, window.offsets_offset, segment_count + 1);
                output_span = elements(output, window.output_offset, segment_count);

                SushiRuntime::Result<void> fits = require_window(
                    "segmented_reduce", "values", values.size(), value_span.offset,
                    value_span.count);
                if (fits)
                    fits = require_window("segmented_reduce", "segment_offsets",
                                          segment_offsets.size(), offset_span.offset,
                                          offset_span.count);
                if (fits)
                    fits = require_window("segmented_reduce", "output", output.size(),
                                          output_span.offset, output_span.count);
                return fits;
            }

            /**
             * @brief Folds segment s of values left to right; an empty segment yields identity.
             * @param offsets Segment s covers values[offsets[s], offsets[s + 1]).
             */
            template <typename T, class Combine>
            [[nodiscard]] inline T fold_segment(const Combine& op, const T* values,
                                                const std::uint32_t* offsets, std::size_t s,
                                                T identity)
            {
                const std::size_t begin = offsets[s];
                const std::size_t end = offsets[s + 1];
                return (end > begin) ? fold_tile(op, values, begin, end) : identity;
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
        template <typename T, class Combine>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(SushiRuntime::Recorder& recorder,
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

            SushiRuntime::Result<void> named = Detail::require_options("segmented_reduce", options);
            if (!named)
                return named;
            Detail::NodeStamper stamper(options, "graph.segmented_reduce");

            Detail::ElementSpan<T> value_span;
            Detail::ElementSpan<std::uint32_t> offset_span;
            Detail::ElementSpan<T> output_span;
            SushiRuntime::Result<void> fits =
                Detail::segment_spans(values, segment_offsets, output, segment_count, window,
                                      value_span, offset_span, output_span);
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
        template <typename T, class Combine>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(SushiRuntime::Recorder& recorder,
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

            SushiRuntime::Result<void> named = Detail::require_options("segmented_reduce", options);
            if (!named)
                return named;
            Detail::NodeStamper stamper(options, "graph.segmented_reduce");

            const std::size_t capacity = segment_count.capacity();
            Detail::ElementSpan<T> value_span;
            Detail::ElementSpan<std::uint32_t> offset_span;
            Detail::ElementSpan<T> output_span;
            SushiRuntime::Result<void> fits =
                Detail::segment_spans(values, segment_offsets, output, capacity, window,
                                      value_span, offset_span, output_span);
            if (!fits)
                return fits;

            const T* value_base = value_span.data();
            const std::uint32_t* offset_base = offset_span.data();
            T* out_base = output_span.data();

            // state[0] is the live segment count the launch reads on the device.
            auto state_block = recorder.scratch<std::uint32_t>(
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
        template <typename T, class Combine>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(SushiRuntime::Recorder& recorder,
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
        template <typename T, class Combine>
        [[nodiscard]] SushiRuntime::Result<void>
        segmented_reduce(SushiRuntime::Recorder& recorder,
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
