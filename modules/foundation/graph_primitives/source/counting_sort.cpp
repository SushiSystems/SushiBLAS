/**************************************************************************/
/* counting_sort.cpp                                                      */
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
 * @file counting_sort.cpp
 * @brief Defines counting_sort(), compiled once for the device.
 * @author Mustafa Garip
 */

#include <SushiBLAS/graph_primitives/counting_sort.hpp>

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
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>
#include <SushiBLAS/graph_primitives/node_stamper.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include <SushiBLAS/graph_primitives/radix_plan.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        SushiRuntime::Result<void>
        counting_sort(SushiRuntime::Recorder& recorder,
                      const SushiRuntime::Buffer<std::uint32_t>& keys,
                      const SushiRuntime::Buffer<std::uint32_t>& indices,
                      const SushiRuntime::Buffer<std::uint32_t>& offsets,
                      const SushiRuntime::Buffer<std::uint32_t>& total,
                      SushiRuntime::LateCount n,
                      SushiRuntime::LateCount key_range,
                      const SushiRuntime::Buffer<std::uint32_t>* counts,
                      CountingSortWindow window,
                      const std::optional<PrimitiveOptions>& options)
        {
            using SushiRuntime::Access;
            using SushiRuntime::DeviceSized;
            using SushiRuntime::LateCount;
            using SushiRuntime::Reads;
            using SushiRuntime::Writes;
            using SushiRuntime::BufferRegion;

            SushiRuntime::Result<void> named = Detail::require_options("counting_sort", options);
            if (!named)
                return named;
            Detail::NodeStamper stamper(options, "graph.counting_sort");

            const std::size_t n_capacity = n.capacity();
            const std::size_t key_capacity = key_range.capacity();

            if (key_capacity == 0)
                return Detail::invalid_graph("counting_sort: key_range.capacity() is zero.");

            using Span = Detail::ElementSpan<std::uint32_t>;
            const Span keys_span = Detail::elements(keys, window.keys_offset, n_capacity);
            const Span indices_span = Detail::elements(indices, window.indices_offset, n_capacity);
            const Span offsets_span =
                Detail::elements(offsets, window.offsets_offset, key_capacity + 1);
            const Span total_span = Detail::elements(total, window.total_offset, std::size_t{1});
            const Span counts_span = (counts != nullptr)
                ? Detail::elements(*counts, window.counts_offset, key_capacity)
                : Span{};

            SushiRuntime::Result<void> fits = Detail::require_window(
                "counting_sort", "keys", keys.size(), keys_span.offset, keys_span.count);
            if (fits)
                fits = Detail::require_window("counting_sort", "indices", indices.size(),
                                              indices_span.offset, indices_span.count);
            if (fits)
                fits = Detail::require_window("counting_sort", "offsets", offsets.size(),
                                              offsets_span.offset, offsets_span.count);
            if (fits)
                fits = Detail::require_window("counting_sort", "total", total.size(),
                                              total_span.offset, total_span.count);
            if (fits && counts != nullptr)
                fits = Detail::require_window("counting_sort", "counts", counts->size(),
                                              counts_span.offset, counts_span.count);
            if (!fits)
                return fits;
            if (indices.device_index() != keys.device_index() ||
                offsets.device_index() != keys.device_index() ||
                (counts != nullptr && counts->device_index() != keys.device_index()))
                return Detail::invalid_graph(
                    "counting_sort: keys, indices, offsets and counts must live on the same "
                    "device.");

            const std::size_t digit_count = Detail::radix_digit_count(key_capacity);
            const std::size_t plan_words = digit_count * Detail::K_RADIX_PLAN_WORDS;
            const auto words = [](std::uint32_t* base, std::size_t count)
            {
                return BufferRegion{static_cast<void*>(base), 0, count * sizeof(std::uint32_t)};
            };

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            // n_state[0] is the live element count, n_state[1] its live tile count.
            auto n_state_block = recorder.scratch<std::uint32_t>(
                2, keys.device_index(), keys.residency());
            if (!n_state_block)
                return SushiRuntime::make_error(std::move(n_state_block.error()));
            std::uint32_t* n_state = *n_state_block;
            const BufferRegion n_state_region = words(n_state, 2);

            // The plan holds, per digit, its live work counts and where its placement lands.
            std::uint32_t* plan = nullptr;
            if (n_capacity != 0)
            {
                auto plan_block = recorder.scratch<std::uint32_t>(
                    plan_words, keys.device_index(), keys.residency());
                if (!plan_block)
                    return SushiRuntime::make_error(std::move(plan_block.error()));
                plan = *plan_block;
            }
            const BufferRegion plan_region = words(plan, plan_words);

            // One node resolves both counts, publishes total and fills the digit plan.
            std::uint32_t* total_base = total_span.data();
            recorder.resolve(std::array<LateCount, 2>{n, key_range}, keys.device_index(),
                Access(Reads(), (plan != nullptr)
                                    ? Detail::access_of<Writes>(n_state_region, total_span,
                                                                plan_region)
                                    : Detail::access_of<Writes>(n_state_region, total_span)),
                [n_state, total_base, plan, digit_count](const std::array<std::size_t, 2>& live)
                {
                    const std::size_t tiles = Detail::next_live_tile_count(live[0]);
                    n_state[0] = static_cast<std::uint32_t>(live[0]);
                    n_state[1] = static_cast<std::uint32_t>(tiles);
                    total_base[0] = static_cast<std::uint32_t>(live[0]);
                    if (plan != nullptr)
                        Detail::fill_radix_plan(digit_count, live[0], tiles, live[1], plan);
                },
                stamper.stage("resolve", std::size_t{1}));

            std::uint32_t* offsets_base = offsets_span.data();
            std::uint32_t* counts_base = (counts != nullptr) ? counts_span.data() : nullptr;

            if (n_capacity == 0)
            {
                // No element ever arrives, so every bucket starts and stays at zero.
                recorder.kernel(
                    DeviceSized{key_capacity + 1, nullptr, nullptr},
                    Access(Reads(), (counts != nullptr)
                                        ? Detail::access_of<Writes>(offsets_span, counts_span)
                                        : Detail::access_of<Writes>(offsets_span)),
                    [offsets_base, counts_base, key_capacity](std::size_t k)
                    {
                        offsets_base[k] = 0;
                        if (counts_base != nullptr && k < key_capacity)
                            counts_base[k] = 0;
                    },
                    stamper.stage("offsets", key_capacity + 1));
                return Detail::recording_outcome(recorder);
            }

            const std::size_t tile_capacity = Detail::next_live_tile_count(n_capacity);

            // hist holds raw counts until the column scan overwrites it in place with prefixes.
            auto hist_block = recorder.scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS * tile_capacity, keys.device_index(), keys.residency());
            if (!hist_block)
                return SushiRuntime::make_error(std::move(hist_block.error()));
            std::uint32_t* hist = *hist_block;
            auto column_total_block = recorder.scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS, keys.device_index(), keys.residency());
            if (!column_total_block)
                return SushiRuntime::make_error(std::move(column_total_block.error()));
            std::uint32_t* column_total = *column_total_block;
            auto bin_base_block = recorder.scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS, keys.device_index(), keys.residency());
            if (!bin_base_block)
                return SushiRuntime::make_error(std::move(bin_base_block.error()));
            std::uint32_t* bin_base = *bin_base_block;

            // The digits alternate between indices and this buffer; one digit never needs it.
            std::uint32_t* alternate = nullptr;
            if (digit_count >= 2)
            {
                auto alternate_block = recorder.scratch<std::uint32_t>(
                    n_capacity, keys.device_index(), keys.residency());
                if (!alternate_block)
                    return SushiRuntime::make_error(std::move(alternate_block.error()));
                alternate = *alternate_block;
            }

            const std::uint32_t* keys_base = keys_span.data();
            std::uint32_t* indices_base = indices_span.data();

            const BufferRegion indices_region = Detail::region_of(indices_span);
            const BufferRegion alternate_region = words(alternate, n_capacity);
            const BufferRegion hist_region =
                words(hist, Detail::K_RADIX_DIGIT_BINS * tile_capacity);
            const BufferRegion column_total_region =
                words(column_total, Detail::K_RADIX_DIGIT_BINS);
            const BufferRegion bin_base_region = words(bin_base, Detail::K_RADIX_DIGIT_BINS);

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
                    Reads reads = Detail::access_of<Reads>(keys_span, plan_region);
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
                        stamper.stage("histogram", tile_capacity));
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
                    stamper.stage("column_scan", Detail::K_RADIX_DIGIT_BINS));

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
                    stamper.stage("bin_base", std::size_t{1}));

                // Placement: tiles write in ascending index order, so the sort is stable.
                {
                    Reads reads = Detail::access_of<Reads>(keys_span, hist_region,
                                                           bin_base_region, plan_region);
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
                        stamper.stage("placement", tile_capacity));
                }
            }

            // Offsets: one lower bound per bucket over the sorted keys; no live range is read.
            recorder.kernel(
                DeviceSized{key_capacity + 1, nullptr, n_state},
                Access(Detail::access_of<Reads>(keys_span, indices_span, n_state_region),
                       Detail::access_of<Writes>(offsets_span)),
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
                stamper.stage("offsets", key_capacity + 1));

            if (counts == nullptr)
                return Detail::recording_outcome(recorder);

            // Counts are differences of two settled offsets and need no live scalar.
            recorder.kernel(
                DeviceSized{key_capacity, nullptr, nullptr},
                Access(Detail::access_of<Reads>(offsets_span),
                       Detail::access_of<Writes>(counts_span)),
                [offsets_base, counts_base](std::size_t k)
                {
                    counts_base[k] = offsets_base[k + 1] - offsets_base[k];
                },
                stamper.stage("counts", key_capacity));

            return Detail::recording_outcome(recorder);
        }
    } // namespace GraphPrimitives
} // namespace SushiBLAS
