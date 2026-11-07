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
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>

#include <SushiBLAS/graph_primitives/argument_checks.hpp>
#include <SushiBLAS/graph_primitives/element_window.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>
#include <SushiBLAS/graph_primitives/node_stamper.hpp>
#include <SushiBLAS/graph_primitives/primitive_options.hpp>
#include <SushiBLAS/graph_primitives/primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/radix_plan.hpp>
#include <SushiBLAS/graph_primitives/runtime_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/sort_resolution.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        template <class Port>
        SushiRuntime::Result<void>
        counting_sort(Port& recorder,
                      const typename Port::template Buffer<std::uint32_t>& keys,
                      const typename Port::template Buffer<std::uint32_t>& indices,
                      const typename Port::template Buffer<std::uint32_t>& offsets,
                      const typename Port::template Buffer<std::uint32_t>& total,
                      SushiRuntime::LateCount n,
                      SushiRuntime::LateCount key_range,
                      const typename Port::template Buffer<std::uint32_t>* counts,
                      CountingSortWindow window,
                      const std::optional<PrimitiveOptions>& options)
        {
            static_assert(K_IS_PRIMITIVE_RECORDER<Port>, "Port must be a PrimitiveRecorder.");
            using SushiRuntime::Access;
            using SushiRuntime::DeviceSized;
            using SushiRuntime::LateCount;
            using SushiRuntime::Reads;
            using SushiRuntime::Writes;
            using SushiRuntime::BufferRegion;

            const std::size_t n_capacity = n.capacity();
            const std::size_t key_capacity = key_range.capacity();

            using Span = Detail::ElementSpan<typename Port::template Buffer<std::uint32_t>>;
            const Span keys_span = Detail::elements(keys, window.keys_offset, n_capacity);
            const Span indices_span = Detail::elements(indices, window.indices_offset, n_capacity);
            const Span offsets_span =
                Detail::elements(offsets, window.offsets_offset, key_capacity + 1);
            const Span total_span = Detail::elements(total, window.total_offset, std::size_t{1});
            const Span counts_span = (counts != nullptr)
                ? Detail::elements(*counts, window.counts_offset, key_capacity)
                : Span{};

            Detail::CountingSortShape shape;
            shape.n_capacity = n_capacity;
            shape.key_capacity = key_capacity;
            shape.keys = Detail::Operand{keys.size(), window.keys_offset, keys.device_index()};
            shape.indices =
                Detail::Operand{indices.size(), window.indices_offset, indices.device_index()};
            shape.offsets =
                Detail::Operand{offsets.size(), window.offsets_offset, offsets.device_index()};
            shape.total = Detail::Operand{total.size(), window.total_offset, total.device_index()};
            if (counts != nullptr)
                shape.counts =
                    Detail::Operand{counts->size(), window.counts_offset, counts->device_index()};
            SushiRuntime::Result<void> checked = Detail::validate_counting_sort(shape, options);
            if (!checked)
                return checked;
            Detail::NodeStamper stamper(options, "graph.counting_sort");

            const std::size_t digit_count = Detail::radix_digit_count(key_capacity);
            const std::size_t plan_words = digit_count * Detail::K_RADIX_PLAN_WORDS;
            const auto words = [](std::uint32_t* base, std::size_t count)
            {
                return BufferRegion{static_cast<void*>(base), 0, count * sizeof(std::uint32_t)};
            };

            // Every node this call records belongs to one rule.
            auto rule = recorder.rule();

            // n_state[0] is the live element count, n_state[1] its live tile count.
            auto n_state_block = recorder.template scratch<std::uint32_t>(
                2, keys.device_index(), keys.residency());
            if (!n_state_block)
                return SushiRuntime::make_error(std::move(n_state_block.error()));
            std::uint32_t* n_state = *n_state_block;
            const BufferRegion n_state_region = words(n_state, 2);

            // The plan holds, per digit, its live work counts and where its placement lands.
            std::uint32_t* plan = nullptr;
            if (n_capacity != 0)
            {
                auto plan_block = recorder.template scratch<std::uint32_t>(
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
                    Detail::publish_sort_state(digit_count, live[0], live[1], n_state,
                                               total_base, plan);
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
                        Detail::clear_empty_bucket(k, key_capacity, offsets_base, counts_base);
                    },
                    stamper.stage("offsets", key_capacity + 1));
                return Detail::recording_outcome(recorder);
            }

            const std::size_t tile_capacity = Detail::next_live_tile_count(n_capacity);

            // hist holds raw counts until the column scan overwrites it in place with prefixes.
            auto hist_block = recorder.template scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS * tile_capacity, keys.device_index(), keys.residency());
            if (!hist_block)
                return SushiRuntime::make_error(std::move(hist_block.error()));
            std::uint32_t* hist = *hist_block;
            auto column_total_block = recorder.template scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS, keys.device_index(), keys.residency());
            if (!column_total_block)
                return SushiRuntime::make_error(std::move(column_total_block.error()));
            std::uint32_t* column_total = *column_total_block;
            auto bin_base_block = recorder.template scratch<std::uint32_t>(
                Detail::K_RADIX_DIGIT_BINS, keys.device_index(), keys.residency());
            if (!bin_base_block)
                return SushiRuntime::make_error(std::move(bin_base_block.error()));
            std::uint32_t* bin_base = *bin_base_block;

            // The digits alternate between indices and this buffer; one digit never needs it.
            std::uint32_t* alternate = nullptr;
            if (digit_count >= 2)
            {
                auto alternate_block = recorder.template scratch<std::uint32_t>(
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
                    offsets_base[k] = Detail::bucket_start(keys_base, indices_base, live_n, k);
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

        template SushiRuntime::Result<void>
        counting_sort<RuntimePrimitiveRecorder>(
            RuntimePrimitiveRecorder&,
            const RuntimePrimitiveRecorder::Buffer<std::uint32_t>&,
            const RuntimePrimitiveRecorder::Buffer<std::uint32_t>&,
            const RuntimePrimitiveRecorder::Buffer<std::uint32_t>&,
            const RuntimePrimitiveRecorder::Buffer<std::uint32_t>&,
            SushiRuntime::LateCount, SushiRuntime::LateCount,
            const RuntimePrimitiveRecorder::Buffer<std::uint32_t>*, CountingSortWindow,
            const std::optional<PrimitiveOptions>&);

        template SushiRuntime::Result<void>
        counting_sort<HostPrimitiveRecorder>(
            HostPrimitiveRecorder&,
            const HostPrimitiveRecorder::Buffer<std::uint32_t>&,
            const HostPrimitiveRecorder::Buffer<std::uint32_t>&,
            const HostPrimitiveRecorder::Buffer<std::uint32_t>&,
            const HostPrimitiveRecorder::Buffer<std::uint32_t>&,
            SushiRuntime::LateCount, SushiRuntime::LateCount,
            const HostPrimitiveRecorder::Buffer<std::uint32_t>*, CountingSortWindow,
            const std::optional<PrimitiveOptions>&);
    } // namespace GraphPrimitives
} // namespace SushiBLAS
