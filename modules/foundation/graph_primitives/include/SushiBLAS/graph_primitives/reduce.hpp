/**************************************************************************/
/* reduce.hpp                                                             */
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
 * @file reduce.hpp
 * @brief Declares reduce(), a fixed-order reduction recorded through a Recorder.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/device_sized.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>

#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Records a reduction of the first n values whose combination order never varies.
         * @tparam Combine Binary functor; must be trivially copyable.
         * @param output Receives the result in output[0]; identity when n is zero.
         * @throws SushiRuntime::SushiException With invalid_graph if n exceeds the size
         *         of input or output is empty, or allocation_failed if scratch does not fit.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <typename T, class Combine>
        void reduce(SushiRuntime::Recorder& recorder,
                    const SushiRuntime::Buffer<T>& input,
                    const SushiRuntime::Buffer<T>& output, std::size_t n, Combine op,
                    T identity)
        {
            using SushiRuntime::Access;
            using SushiRuntime::NodeOptions;
            using SushiRuntime::Reads;
            using SushiRuntime::Writes;
            using SushiRuntime::BufferRegion;

            if (n > input.size())
                Detail::throw_invalid_graph(
                    "reduce: n (" + std::to_string(n) + ") exceeds the input buffer's size (" +
                    std::to_string(input.size()) + ").");
            if (output.size() == 0)
                Detail::throw_invalid_graph("reduce: the output buffer holds no elements.");

            T* out_base = output.data();
            if (n == 0)
            {
                // An empty reduction still records a node, so readers order after it.
                recorder.kernel(std::size_t{1}, Access(Reads(), Writes(output)),
                    [out_base, identity](std::size_t)
                    {
                        out_base[0] = identity;
                    },
                    NodeOptions{std::size_t{1}});
                return;
            }

            const std::vector<Detail::ReduceLevel> levels = Detail::plan_reduction(n);
            const std::size_t scratch_count = Detail::reduce_scratch_count(levels);

            // A single level writes the output directly and takes no scratch.
            T* scratch = nullptr;
            if (scratch_count != 0)
                scratch = recorder.scratch<T>(scratch_count, input.device_index(),
                                              input.residency());

            const T* in_base = input.data();
            void* scratch_key = static_cast<void*>(scratch);

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            for (std::size_t i = 0; i < levels.size(); ++i)
            {
                const Detail::ReduceLevel& level = levels[i];
                const bool first = (i == 0);
                const bool last = (i + 1 == levels.size());

                const T* src = first ? in_base : (scratch + level.in_offset);
                T* dst = last ? out_base : (scratch + level.out_offset);
                const std::size_t in_count = level.in_count;

                const Reads reads = first
                    ? Reads(input)
                    : Reads(ResourceRegion{scratch_key, level.in_offset * sizeof(T),
                                           level.in_count * sizeof(T)});
                const Writes writes = last
                    ? Writes(output)
                    : Writes(ResourceRegion{scratch_key, level.out_offset * sizeof(T),
                                            level.out_count * sizeof(T)});

                recorder.kernel(level.out_count, Access(reads, writes),
                    [src, dst, in_count, op](std::size_t t)
                    {
                        dst[t] = fold_indexed_tile(op, src, t, in_count);
                    },
                    NodeOptions{level.out_count});
            }
        }

        /**
         * @brief Records a reduction whose live element count is resolved once per run.
         * @tparam Combine Binary functor; must be trivially copyable.
         * @param output Receives the result in output[0]; identity when the live count is zero.
         * @throws SushiRuntime::SushiException With invalid_graph if n.capacity() exceeds
         *         the size of input or output is empty; a run throws if the live count exceeds
         *         n.capacity().
         */
        template <typename T, class Combine>
        void reduce(SushiRuntime::Recorder& recorder,
                    const SushiRuntime::Buffer<T>& input,
                    const SushiRuntime::Buffer<T>& output, SushiRuntime::LateCount n,
                    Combine op, T identity)
        {
            using SushiRuntime::Access;
            using SushiRuntime::DeviceSized;
            using SushiRuntime::LateCount;
            using SushiRuntime::NodeOptions;
            using SushiRuntime::Reads;
            using SushiRuntime::Writes;
            using SushiRuntime::BufferRegion;

            const std::size_t capacity = n.capacity();
            if (capacity > input.size())
                Detail::throw_invalid_graph(
                    "reduce: n.capacity() (" + std::to_string(capacity) +
                    ") exceeds the input buffer's size (" + std::to_string(input.size()) + ").");
            if (output.size() == 0)
                Detail::throw_invalid_graph("reduce: the output buffer holds no elements.");

            const std::vector<Detail::ReduceLevel> levels = Detail::plan_tiled_levels(capacity);
            T* out_base = output.data();

            if (levels.empty())
            {
                // With capacity zero the identity is the answer on every run.
                recorder.kernel(std::size_t{1}, Access(Reads(), Writes(output)),
                    [out_base, identity](std::size_t)
                    {
                        out_base[0] = identity;
                    });
                return;
            }

            const std::size_t scratch_count = Detail::reduce_scratch_count(levels);
            T* scratch = nullptr;
            if (scratch_count != 0)
                scratch = recorder.scratch<T>(scratch_count, input.device_index(),
                                              input.residency());

            // state[i] is the live in_count of level i, state[level_count + i] its out_count.
            const std::size_t level_count = levels.size();
            std::uint32_t* state = recorder.scratch<std::uint32_t>(
                2 * level_count, input.device_index(), input.residency());
            const ResourceRegion state_region{state, 0, 2 * level_count * sizeof(std::uint32_t)};

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            recorder.resolve(std::array<LateCount, 1>{n}, input.device_index(),
                Access(Reads(), Writes(state_region)),
                [state, level_count](const std::array<std::size_t, 1>& live)
                {
                    Detail::fill_live_level_words(level_count, live[0], state);
                },
                NodeOptions{std::size_t{1}});

            const T* in_base = input.data();
            void* scratch_key = static_cast<void*>(scratch);

            for (std::size_t i = 0; i < levels.size(); ++i)
            {
                const Detail::ReduceLevel& level = levels[i];
                const bool first = (i == 0);
                const bool last = (i + 1 == levels.size());

                const T* src = first ? in_base : (scratch + level.in_offset);
                T* dst = last ? out_base : (scratch + level.out_offset);

                const Reads reads = first
                    ? Reads(input, state_region)
                    : Reads(ResourceRegion{scratch_key, level.in_offset * sizeof(T),
                                           level.in_count * sizeof(T)},
                            state_region);
                const Writes writes = last
                    ? Writes(output)
                    : Writes(ResourceRegion{scratch_key, level.out_offset * sizeof(T),
                                            level.out_count * sizeof(T)});

                recorder.kernel(
                    DeviceSized{level.out_count, state + level_count + i, state + i},
                    Access(reads, writes),
                    [src, dst, op](std::size_t t, std::size_t in_count)
                    {
                        dst[t] = fold_indexed_tile(op, src, t, in_count);
                    },
                    NodeOptions{level.out_count});
            }

            // A live count of zero folds no level, so this node is the only writer of identity.
            recorder.kernel(
                DeviceSized{std::size_t{1}, nullptr, state},
                Access(Reads(state_region), Writes(output)),
                [out_base, identity](std::size_t, std::size_t live_n)
                {
                    if (live_n == 0)
                        out_base[0] = identity;
                },
                NodeOptions{std::size_t{1}});
        }

        /**
         * @brief Records a fixed-count reduction that takes identity from Combine::identity().
         * @tparam Combine Binary functor with a static identity(); must be trivially copyable.
         * @throws SushiRuntime::SushiException As the overload that takes an identity.
         */
        template <typename T, class Combine>
        void reduce(SushiRuntime::Recorder& recorder,
                    const SushiRuntime::Buffer<T>& input,
                    const SushiRuntime::Buffer<T>& output, std::size_t n, Combine op)
        {
            GraphPrimitives::reduce(recorder, input, output, n, op, Combine::identity());
        }

        /**
         * @brief Records a late-bound reduction that takes identity from Combine::identity().
         * @tparam Combine Binary functor with a static identity(); must be trivially copyable.
         * @throws SushiRuntime::SushiException As the overload that takes an identity.
         */
        template <typename T, class Combine>
        void reduce(SushiRuntime::Recorder& recorder,
                    const SushiRuntime::Buffer<T>& input,
                    const SushiRuntime::Buffer<T>& output, SushiRuntime::LateCount n,
                    Combine op)
        {
            GraphPrimitives::reduce(recorder, input, output, n, op, Combine::identity());
        }

    } // namespace GraphPrimitives
} // namespace SushiBLAS
