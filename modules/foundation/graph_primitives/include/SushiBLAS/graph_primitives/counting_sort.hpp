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
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>
#include <SushiBLAS/graph_primitives/radix_plan.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Records a stable argsort of uint32 keys by radix digits, with bucket offsets.
         * @param indices Receives the sorted order; holds n.capacity() elements.
         * @param offsets Receives the bucket starts; holds key_range.capacity() + 1 elements.
         * @param counts  Optional bucket sizes; holds key_range.capacity() elements.
         * @throws SushiRuntime::Core::SushiException With invalid_graph for a broken size rule.
         * @see modules/foundation/graph_primitives/README.md
         */
        inline void counting_sort(SushiRuntime::API::Recorder& recorder,
                                  const SushiRuntime::API::Buffer<std::uint32_t>& keys,
                                  const SushiRuntime::API::Buffer<std::uint32_t>& indices,
                                  const SushiRuntime::API::Buffer<std::uint32_t>& offsets,
                                  const SushiRuntime::API::Buffer<std::uint32_t>& total,
                                  SushiRuntime::API::LateCount n,
                                  SushiRuntime::API::LateCount key_range,
                                  const SushiRuntime::API::Buffer<std::uint32_t>* counts = nullptr)
        {
            using SushiRuntime::API::Access;
            using SushiRuntime::API::DeviceSized;
            using SushiRuntime::API::LateCount;
            using SushiRuntime::API::NodeOptions;
            using SushiRuntime::API::Reads;
            using SushiRuntime::API::Writes;
            using SushiRuntime::Memory::ResourceRegion;

            const std::size_t n_capacity = n.capacity();
            const std::size_t key_capacity = key_range.capacity();

            if (key_capacity == 0)
                Detail::throw_invalid_graph("counting_sort: key_range.capacity() is zero.");
            if (n_capacity > keys.size())
                Detail::throw_invalid_graph(
                    "counting_sort: n.capacity() (" + std::to_string(n_capacity) +
                    ") exceeds the keys buffer's size (" + std::to_string(keys.size()) + ").");
            if (indices.size() < n_capacity)
                Detail::throw_invalid_graph(
                    "counting_sort: indices holds " + std::to_string(indices.size()) +
                    " elements but " + std::to_string(n_capacity) + " are required.");
            if (offsets.size() < key_capacity + 1)
                Detail::throw_invalid_graph(
                    "counting_sort: offsets holds " + std::to_string(offsets.size()) +
                    " elements but " + std::to_string(key_capacity + 1) + " are required.");
            if (total.size() == 0)
                Detail::throw_invalid_graph("counting_sort: the total buffer holds no elements.");
            if (counts != nullptr && counts->size() < key_capacity)
                Detail::throw_invalid_graph(
                    "counting_sort: counts holds " + std::to_string(counts->size()) +
                    " elements but " + std::to_string(key_capacity) + " are required.");
            if (indices.device_index() != keys.device_index() ||
                offsets.device_index() != keys.device_index() ||
                (counts != nullptr && counts->device_index() != keys.device_index()))
                Detail::throw_invalid_graph(
                    "counting_sort: keys, indices, offsets and counts must live on the same "
                    "device.");

            const std::size_t digit_count = Detail::radix_digit_count(key_capacity);
            const std::size_t plan_words = digit_count * Detail::K_RADIX_PLAN_WORDS;
            const auto words = [](std::uint32_t* base, std::size_t count)
            {
                return ResourceRegion{static_cast<void*>(base), 0, count * sizeof(std::uint32_t)};
            };

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            // n_state[0] is the live element count, n_state[1] its live tile count.
            std::uint32_t* n_state = recorder.scratch<std::uint32_t>(
                2, keys.device_index(), keys.residency());
            const ResourceRegion n_state_region = words(n_state, 2);

            // The plan holds, per digit, its live work counts and where its placement lands.
            std::uint32_t* plan = nullptr;
            if (n_capacity != 0)
                plan = recorder.scratch<std::uint32_t>(plan_words, keys.device_index(),
                                                       keys.residency());
            const ResourceRegion plan_region = words(plan, plan_words);

            // One node resolves both counts, publishes total and fills the digit plan.
            std::uint32_t* total_base = total.data();
            recorder.resolve(std::array<LateCount, 2>{n, key_range}, keys.device_index(),
                Access(Reads(), (plan != nullptr) ? Writes(n_state_region, total, plan_region)
                                                  : Writes(n_state_region, total)),
                [n_state, total_base, plan, digit_count](const std::array<std::size_t, 2>& live)
                {
                    const std::size_t tiles = Detail::next_live_tile_count(live[0]);
                    n_state[0] = static_cast<std::uint32_t>(live[0]);
                    n_state[1] = static_cast<std::uint32_t>(tiles);
                    total_base[0] = static_cast<std::uint32_t>(live[0]);
                    if (plan != nullptr)
                        Detail::fill_radix_plan(digit_count, live[0], tiles, live[1], plan);
                },
                NodeOptions{std::size_t{1}});

            std::uint32_t* offsets_base = offsets.data();
            std::uint32_t* counts_base = (counts != nullptr) ? counts->data() : nullptr;

            if (n_capacity == 0)
            {
                // No element ever arrives, so every bucket starts and stays at zero.
                recorder.kernel(
                    DeviceSized{key_capacity + 1, nullptr, nullptr},
                    Access(Reads(), (counts != nullptr) ? Writes(offsets, *counts)
                                                        : Writes(offsets)),
                    [offsets_base, counts_base, key_capacity](std::size_t k)
                    {
                        offsets_base[k] = 0;
                        if (counts_base != nullptr && k < key_capacity)
                            counts_base[k] = 0;
                    },
                    NodeOptions{key_capacity + 1});
                return;
            }

            const std::size_t tile_capacity = Detail::next_live_tile_count(n_capacity);

            // hist holds raw counts until the column scan overwrites it in place with prefixes.
            std::uint32_t* hist = recorder.scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS * tile_capacity, keys.device_index(),
                keys.residency());
            std::uint32_t* column_total = recorder.scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS, keys.device_index(), keys.residency());
            std::uint32_t* bin_base = recorder.scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS, keys.device_index(), keys.residency());

            // The digits alternate between indices and this buffer; one digit never needs it.
            std::uint32_t* alternate = nullptr;
            if (digit_count >= 2)
                alternate = recorder.scratch<std::uint32_t>(n_capacity, keys.device_index(),
                                                            keys.residency());

            const std::uint32_t* keys_base = keys.data();
            std::uint32_t* indices_base = indices.data();

            const ResourceRegion indices_region = words(indices_base, n_capacity);
            const ResourceRegion alternate_region = words(alternate, n_capacity);
            const ResourceRegion hist_region =
                words(hist, Detail::K_RADIX_DIGIT_BINS * tile_capacity);
            const ResourceRegion column_total_region =
                words(column_total, Detail::K_RADIX_DIGIT_BINS);
            const ResourceRegion bin_base_region = words(bin_base, Detail::K_RADIX_DIGIT_BINS);

            for (std::size_t d = 0; d < digit_count; ++d)
            {
                const bool first = (d == 0);
                const auto shift = static_cast<std::uint32_t>(Detail::K_RADIX_DIGIT_BITS * d);
                const std::uint32_t* digit_plan = plan + d * Detail::K_RADIX_PLAN_WORDS;
                const std::uint32_t* live_tiles = digit_plan + Detail::K_RADIX_PLAN_TILES;
                const std::uint32_t* snapshot = digit_plan + Detail::K_RADIX_PLAN_LIVE_N;

                // A later digit reads whichever buffer the digit before it filled.
                const auto add_sources = [&](Reads& reads)
                {
                    if (first)
                        return;
                    reads.keys.push_back(indices_region);
                    if (alternate != nullptr)
                        reads.keys.push_back(alternate_region);
                };

                // Histogram: each tile overwrites its own row, so no run clears hist.
                {
                    Reads reads(keys, plan_region);
                    add_sources(reads);
                    recorder.kernel(
                        DeviceSized{tile_capacity, live_tiles, snapshot},
                        Access(reads, Writes(hist_region)),
                        [keys_base, indices_base, alternate, hist, first, shift, tile_capacity](
                            std::size_t t, std::size_t live_n, std::size_t into_indices)
                        {
                            const std::uint32_t* source =
                                into_indices != 0 ? alternate : indices_base;
                            std::uint32_t local[Detail::K_RADIX_DIGIT_BINS] = {};
                            const std::size_t begin = t * K_REDUCE_TILE;
                            std::size_t end = begin + K_REDUCE_TILE;
                            if (end > live_n)
                                end = live_n;
                            for (std::size_t i = begin; i < end; ++i)
                            {
                                const std::size_t element = first ? i : source[i];
                                ++local[Detail::radix_digit(keys_base[element], shift)];
                            }
                            for (std::size_t b = 0; b < Detail::K_RADIX_DIGIT_BINS; ++b)
                                hist[b * tile_capacity + t] = local[b];
                        },
                        NodeOptions{tile_capacity});
                }

                // Column scan: exactly one work-item touches each bin's column of hist.
                recorder.kernel(
                    DeviceSized{Detail::K_RADIX_DIGIT_BINS,
                                digit_plan + Detail::K_RADIX_PLAN_BINS, n_state + 1},
                    Access(Reads(hist_region, n_state_region, plan_region),
                           Writes(hist_region, column_total_region)),
                    [hist, column_total, tile_capacity](std::size_t b, std::size_t live_tile_count)
                    {
                        std::uint32_t running = 0;
                        for (std::size_t t = 0; t < live_tile_count; ++t)
                        {
                            const std::uint32_t count = hist[b * tile_capacity + t];
                            hist[b * tile_capacity + t] = running;
                            running += count;
                        }
                        column_total[b] = running;
                    },
                    NodeOptions{Detail::K_RADIX_DIGIT_BINS});

                // Bin base: one work-item walks the bin totals, whatever n is.
                recorder.kernel(
                    DeviceSized{std::size_t{1}, digit_plan + Detail::K_RADIX_PLAN_ONE, nullptr},
                    Access(Reads(column_total_region, plan_region), Writes(bin_base_region)),
                    [column_total, bin_base](std::size_t)
                    {
                        std::uint32_t running = 0;
                        for (std::size_t b = 0; b < Detail::K_RADIX_DIGIT_BINS; ++b)
                        {
                            bin_base[b] = running;
                            running += column_total[b];
                        }
                    },
                    NodeOptions{std::size_t{1}});

                // Placement: tiles write in ascending index order, so the sort is stable.
                {
                    Reads reads(keys, hist_region, bin_base_region, plan_region);
                    add_sources(reads);
                    recorder.kernel(
                        DeviceSized{tile_capacity, live_tiles, snapshot},
                        Access(reads, (alternate != nullptr)
                                          ? Writes(indices_region, alternate_region)
                                          : Writes(indices_region)),
                        [keys_base, indices_base, alternate, hist, bin_base, first, shift,
                         tile_capacity](std::size_t t, std::size_t live_n,
                                        std::size_t into_indices)
                        {
                            const std::uint32_t* source =
                                into_indices != 0 ? alternate : indices_base;
                            std::uint32_t* destination =
                                into_indices != 0 ? indices_base : alternate;
                            std::uint32_t cursor[Detail::K_RADIX_DIGIT_BINS];
                            for (std::size_t b = 0; b < Detail::K_RADIX_DIGIT_BINS; ++b)
                                cursor[b] = bin_base[b] + hist[b * tile_capacity + t];
                            const std::size_t begin = t * K_REDUCE_TILE;
                            std::size_t end = begin + K_REDUCE_TILE;
                            if (end > live_n)
                                end = live_n;
                            for (std::size_t i = begin; i < end; ++i)
                            {
                                const std::size_t element = first ? i : source[i];
                                const std::uint32_t digit =
                                    Detail::radix_digit(keys_base[element], shift);
                                destination[cursor[digit]++] =
                                    static_cast<std::uint32_t>(element);
                            }
                        },
                        NodeOptions{tile_capacity});
                }
            }

            // Offsets: one lower bound per bucket over the sorted keys; no live range is read.
            recorder.kernel(
                DeviceSized{key_capacity + 1, nullptr, n_state},
                Access(Reads(keys, indices, n_state_region), Writes(offsets)),
                [keys_base, indices_base, offsets_base](std::size_t k, std::size_t live_n)
                {
                    std::size_t low = 0;
                    std::size_t high = live_n;
                    while (low < high)
                    {
                        const std::size_t middle = low + (high - low) / 2;
                        if (keys_base[indices_base[middle]] < k)
                            low = middle + 1;
                        else
                            high = middle;
                    }
                    offsets_base[k] = static_cast<std::uint32_t>(low);
                },
                NodeOptions{key_capacity + 1});

            if (counts == nullptr)
                return;

            // Counts are differences of two settled offsets and need no live scalar.
            recorder.kernel(
                DeviceSized{key_capacity, nullptr, nullptr},
                Access(Reads(offsets), Writes(*counts)),
                [offsets_base, counts_base](std::size_t k)
                {
                    counts_base[k] = offsets_base[k + 1] - offsets_base[k];
                },
                NodeOptions{key_capacity});
        }

    } // namespace GraphPrimitives
} // namespace SushiBLAS
