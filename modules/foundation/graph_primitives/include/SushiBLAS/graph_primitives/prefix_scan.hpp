/**************************************************************************/
/* prefix_scan.hpp                                                        */
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
 * @file prefix_scan.hpp
 * @brief Declares prefix_scan(), an exclusive scan of uint32 counts into offsets.
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

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Records an exclusive scan of uint32 counts into running offsets.
         * @param counts  Holds at least n.capacity() elements.
         * @param offsets Holds at least n.capacity() + 1; offsets[live n] receives the sum.
         * @param total   Holds at least one element; total[0] receives the sum.
         * @throws SushiRuntime::SushiException With invalid_graph for a short buffer or
         *         for counts and offsets on different devices; a run throws past n.capacity().
         */
        inline void prefix_scan(SushiRuntime::Recorder& recorder,
                                const SushiRuntime::Buffer<std::uint32_t>& counts,
                                const SushiRuntime::Buffer<std::uint32_t>& offsets,
                                const SushiRuntime::Buffer<std::uint32_t>& total,
                                SushiRuntime::LateCount n)
        {
            using SushiRuntime::Access;
            using SushiRuntime::DeviceSized;
            using SushiRuntime::LateCount;
            using SushiRuntime::NodeOptions;
            using SushiRuntime::Reads;
            using SushiRuntime::Writes;
            using SushiRuntime::BufferRegion;

            const std::size_t capacity = n.capacity();
            if (capacity > counts.size())
                Detail::throw_invalid_graph(
                    "prefix_scan: n.capacity() (" + std::to_string(capacity) +
                    ") exceeds the counts buffer's size (" + std::to_string(counts.size()) +
                    ").");
            if (offsets.size() < capacity + 1)
                Detail::throw_invalid_graph(
                    "prefix_scan: offsets holds " + std::to_string(offsets.size()) +
                    " elements but " + std::to_string(capacity + 1) + " are required.");
            if (total.size() == 0)
                Detail::throw_invalid_graph("prefix_scan: the total buffer holds no elements.");
            if (offsets.device_index() != counts.device_index())
                Detail::throw_invalid_graph(
                    "prefix_scan: counts and offsets must live on the same device.");

            const std::vector<Detail::ReduceLevel> levels = Detail::plan_tiled_levels(capacity);
            const std::uint32_t* counts_base = counts.data();
            std::uint32_t* offsets_base = offsets.data();
            std::uint32_t* total_base = total.data();

            if (levels.empty())
            {
                // Capacity zero: the resolver checks the live count and writes an empty scan.
                auto rule = recorder.rule();
                recorder.resolve(std::array<LateCount, 1>{n}, counts.device_index(),
                    Access(Reads(), Writes(offsets, total)),
                    [offsets_base, total_base](const std::array<std::size_t, 1>&)
                    {
                        offsets_base[0] = 0;
                        total_base[0] = 0;
                    },
                    NodeOptions{std::size_t{1}});
                return;
            }

            const std::size_t level_count = levels.size();
            const std::size_t sweep_scratch_count = Detail::reduce_scratch_count(levels);

            // A single level reads counts and writes offsets directly and takes no scratch.
            std::uint32_t* up_scratch = nullptr;
            std::uint32_t* down_scratch = nullptr;
            if (sweep_scratch_count != 0)
            {
                up_scratch = recorder.scratch<std::uint32_t>(
                    sweep_scratch_count, counts.device_index(), counts.residency());
                down_scratch = recorder.scratch<std::uint32_t>(
                    sweep_scratch_count, counts.device_index(), counts.residency());
            }

            // state[i] is the live in_count of level i, state[level_count + i] its out_count.
            const std::size_t state_words = 2 * level_count;
            std::uint32_t* state = recorder.scratch<std::uint32_t>(
                state_words, counts.device_index(), counts.residency());
            const ResourceRegion state_region{state, 0, state_words * sizeof(std::uint32_t)};

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            recorder.resolve(std::array<LateCount, 1>{n}, counts.device_index(),
                Access(Reads(), Writes(state_region)),
                [state, level_count](const std::array<std::size_t, 1>& live)
                {
                    Detail::fill_live_level_words(level_count, live[0], state);
                },
                NodeOptions{std::size_t{1}});

            const auto words = [](std::uint32_t* base, std::size_t offset, std::size_t count)
            {
                return ResourceRegion{static_cast<void*>(base), offset * sizeof(std::uint32_t),
                                      count * sizeof(std::uint32_t)};
            };

            // The up-sweep folds every level but the top into scratch for the down-sweep.
            for (std::size_t i = 0; i + 1 < level_count; ++i)
            {
                const Detail::ReduceLevel& level = levels[i];
                const bool first = (i == 0);

                const std::uint32_t* src = first ? counts_base : (up_scratch + level.in_offset);
                std::uint32_t* dst = up_scratch + level.out_offset;

                const Reads reads = first
                    ? Reads(counts, state_region)
                    : Reads(words(up_scratch, level.in_offset, level.in_count), state_region);
                const Writes writes(words(up_scratch, level.out_offset, level.out_count));

                recorder.kernel(
                    DeviceSized{level.out_count, state + level_count + i, state + i},
                    Access(reads, writes),
                    [src, dst](std::size_t t, std::size_t in_count)
                    {
                        dst[t] = fold_indexed_tile(Sum<std::uint32_t>{}, src, t,
                                                           in_count);
                    },
                    NodeOptions{level.out_count});
            }

            // The down-sweep runs from the top level, which has no level above it, to level 0.
            for (std::size_t i = level_count; i-- > 0;)
            {
                const Detail::ReduceLevel& level = levels[i];
                const bool first = (i == 0);
                const bool top = (i + 1 == level_count);

                const std::uint32_t* src = first ? counts_base : (up_scratch + level.in_offset);
                std::uint32_t* dst = first ? offsets_base : (down_scratch + level.in_offset);

                if (top)
                {
                    const Reads reads = first
                        ? Reads(counts, state_region)
                        : Reads(words(up_scratch, level.in_offset, level.in_count),
                                state_region);
                    const Writes writes = first
                        ? Writes(offsets, total)
                        : Writes(words(down_scratch, level.in_offset, level.in_count), total);

                    // The top pass scans from zero and always runs, so total is published.
                    recorder.kernel(
                        DeviceSized{std::size_t{1}, nullptr, state + i},
                        Access(reads, writes),
                        [src, dst, offsets_base, total_base, first](std::size_t,
                                                                    std::size_t in_count)
                        {
                            std::uint32_t running = 0;
                            for (std::size_t j = 0; j < in_count; ++j)
                            {
                                dst[j] = running;
                                running += src[j];
                            }
                            total_base[0] = running;
                            if (first)
                                offsets_base[in_count] = running;
                        },
                        NodeOptions{std::size_t{1}});
                    continue;
                }

                // The base of each tile is the exclusive prefix the level above wrote for it.
                const std::uint32_t* tile_bases = down_scratch + level.out_offset;
                const ResourceRegion bases_region =
                    words(down_scratch, level.out_offset, level.out_count);

                const Reads reads = first
                    ? Reads(counts, bases_region, state_region)
                    : Reads(words(up_scratch, level.in_offset, level.in_count), bases_region,
                            state_region);
                const Writes writes = first
                    ? Writes(offsets)
                    : Writes(words(down_scratch, level.in_offset, level.in_count));

                recorder.kernel(
                    DeviceSized{level.out_count, state + level_count + i, state + i},
                    Access(reads, writes),
                    [src, dst, tile_bases](std::size_t t, std::size_t in_count)
                    {
                        const std::size_t begin = t * K_REDUCE_TILE;
                        std::size_t end = begin + K_REDUCE_TILE;
                        if (end > in_count)
                            end = in_count;
                        std::uint32_t running = tile_bases[t];
                        for (std::size_t j = begin; j < end; ++j)
                        {
                            dst[j] = running;
                            running += src[j];
                        }
                    },
                    NodeOptions{level.out_count});
            }

            if (level_count == 1)
                return;

            // Level 0 stops at offsets[live n - 1]; this node copies total into offsets[live n].
            recorder.kernel(
                DeviceSized{std::size_t{1}, nullptr, state},
                Access(Reads(total, state_region), Writes(offsets)),
                [offsets_base, total_base](std::size_t, std::size_t live_n)
                {
                    offsets_base[live_n] = total_base[0];
                },
                NodeOptions{std::size_t{1}});
        }

    } // namespace GraphPrimitives
} // namespace SushiBLAS
