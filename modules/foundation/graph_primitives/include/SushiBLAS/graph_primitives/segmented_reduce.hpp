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
#include <string>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/device_sized.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/memory/resource_region.hpp>

#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Throws unless the offsets and the output hold what segment_count needs.
             * @param offsets_size Elements segment_offsets holds; segment_count + 1 are needed.
             * @param output_size  Elements output holds; segment_count are needed.
             * @throws SushiRuntime::Core::SushiException With invalid_graph for a short buffer.
             */
            inline void require_segment_room(std::size_t offsets_size, std::size_t output_size,
                                             std::size_t segment_count)
            {
                if (offsets_size < segment_count + 1)
                    throw_invalid_graph(
                        "segmented_reduce: segment_offsets holds " +
                        std::to_string(offsets_size) + " entries but " +
                        std::to_string(segment_count + 1) + " are required.");
                if (output_size < segment_count)
                    throw_invalid_graph(
                        "segmented_reduce: output holds " + std::to_string(output_size) +
                        " entries but " + std::to_string(segment_count) + " are required.");
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
         * @param segment_offsets segment_count + 1 non-decreasing entries, the last no larger
         *        than the size of values.
         * @param output Receives one result per segment; identity for an empty segment.
         * @throws SushiRuntime::Core::SushiException With invalid_graph for a short buffer.
         */
        template <typename T, class Combine>
        void segmented_reduce(SushiRuntime::API::Recorder& recorder,
                              const SushiRuntime::API::Buffer<T>& values,
                              const SushiRuntime::API::Buffer<std::uint32_t>& segment_offsets,
                              const SushiRuntime::API::Buffer<T>& output,
                              std::size_t segment_count, Combine op, T identity)
        {
            using SushiRuntime::API::Access;
            using SushiRuntime::API::NodeOptions;
            using SushiRuntime::API::Reads;
            using SushiRuntime::API::Writes;

            Detail::require_segment_room(segment_offsets.size(), output.size(), segment_count);

            const T* value_base = values.data();
            const std::uint32_t* offset_base = segment_offsets.data();
            T* out_base = output.data();

            recorder.kernel(segment_count,
                Access(Reads(values, segment_offsets), Writes(output)),
                [value_base, offset_base, out_base, op, identity](std::size_t s)
                {
                    out_base[s] = Detail::fold_segment(op, value_base, offset_base, s, identity);
                },
                NodeOptions{segment_count});
        }

        /**
         * @brief Records a segmented reduction whose segment count is resolved once per run.
         * @tparam Combine Binary functor; must be trivially copyable.
         * @param segment_offsets segment_count.capacity() + 1 non-decreasing entries.
         * @param output segment_count.capacity() entries; those past the live count are kept.
         * @throws SushiRuntime::Core::SushiException With invalid_graph for a short buffer; a
         *         run throws if the live count exceeds the capacity.
         */
        template <typename T, class Combine>
        void segmented_reduce(SushiRuntime::API::Recorder& recorder,
                              const SushiRuntime::API::Buffer<T>& values,
                              const SushiRuntime::API::Buffer<std::uint32_t>& segment_offsets,
                              const SushiRuntime::API::Buffer<T>& output,
                              SushiRuntime::API::LateCount segment_count, Combine op, T identity)
        {
            using SushiRuntime::API::Access;
            using SushiRuntime::API::DeviceSized;
            using SushiRuntime::API::LateCount;
            using SushiRuntime::API::NodeOptions;
            using SushiRuntime::API::Reads;
            using SushiRuntime::API::Writes;
            using SushiRuntime::Memory::ResourceRegion;

            const std::size_t capacity = segment_count.capacity();
            Detail::require_segment_room(segment_offsets.size(), output.size(), capacity);

            const T* value_base = values.data();
            const std::uint32_t* offset_base = segment_offsets.data();
            T* out_base = output.data();

            // state[0] is the live segment count the launch reads on the device.
            std::uint32_t* state = recorder.scratch<std::uint32_t>(
                1, values.device_index(), values.residency());
            const ResourceRegion state_region{state, 0, sizeof(std::uint32_t)};

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            recorder.resolve(std::array<LateCount, 1>{segment_count}, values.device_index(),
                Access(Reads(), Writes(state_region)),
                [state](const std::array<std::size_t, 1>& live)
                {
                    state[0] = static_cast<std::uint32_t>(live[0]);
                },
                NodeOptions{std::size_t{1}});

            recorder.kernel(
                DeviceSized{capacity, state, nullptr},
                Access(Reads(values, segment_offsets, state_region), Writes(output)),
                [value_base, offset_base, out_base, op, identity](std::size_t s)
                {
                    out_base[s] = Detail::fold_segment(op, value_base, offset_base, s, identity);
                },
                NodeOptions{capacity});
        }

        /**
         * @brief Records a fixed-count segmented reduction with Combine::identity().
         * @tparam Combine Binary functor with a static identity(); must be trivially copyable.
         * @throws SushiRuntime::Core::SushiException As the overload that takes an identity.
         */
        template <typename T, class Combine>
        void segmented_reduce(SushiRuntime::API::Recorder& recorder,
                              const SushiRuntime::API::Buffer<T>& values,
                              const SushiRuntime::API::Buffer<std::uint32_t>& segment_offsets,
                              const SushiRuntime::API::Buffer<T>& output,
                              std::size_t segment_count, Combine op)
        {
            GraphPrimitives::segmented_reduce(recorder, values, segment_offsets, output,
                                              segment_count, op, Combine::identity());
        }

        /**
         * @brief Records a late-bound segmented reduction with Combine::identity().
         * @tparam Combine Binary functor with a static identity(); must be trivially copyable.
         * @throws SushiRuntime::Core::SushiException As the overload that takes an identity.
         */
        template <typename T, class Combine>
        void segmented_reduce(SushiRuntime::API::Recorder& recorder,
                              const SushiRuntime::API::Buffer<T>& values,
                              const SushiRuntime::API::Buffer<std::uint32_t>& segment_offsets,
                              const SushiRuntime::API::Buffer<T>& output,
                              SushiRuntime::API::LateCount segment_count, Combine op)
        {
            GraphPrimitives::segmented_reduce(recorder, values, segment_offsets, output,
                                              segment_count, op, Combine::identity());
        }

    } // namespace GraphPrimitives
} // namespace SushiBLAS
