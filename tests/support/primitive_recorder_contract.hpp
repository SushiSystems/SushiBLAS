/**************************************************************************/
/* primitive_recorder_contract.hpp                                        */
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
 * @file primitive_recorder_contract.hpp
 * @brief Declares the checks every PrimitiveRecorder adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/device_sized.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds the work-items the device-sized check launches.
        inline constexpr std::size_t K_CONTRACT_ITEMS = 8;

        /// @brief Holds the live width the device-sized check publishes.
        inline constexpr std::uint32_t K_CONTRACT_LIVE_WIDTH = 3;

        /// @brief Holds the live count the late-count check resolves.
        inline constexpr std::size_t K_CONTRACT_LIVE_COUNT = 5;

        /** @brief Holds the checks that define a PrimitiveRecorder; an adapter suite calls each one. */
        struct PrimitiveRecorderContract
        {
            /** @brief Returns the region of count words starting at words. */
            [[nodiscard]] static SushiRuntime::BufferRegion
            region_of(const std::uint32_t* words, std::size_t count)
            {
                return SushiRuntime::BufferRegion{const_cast<std::uint32_t*>(words), 0,
                                                  count * sizeof(std::uint32_t)};
            }

            /** @brief Records a node that writes region and runs body once. */
            template <typename Subject, typename Body>
            static void write_once(Subject& subject, const SushiRuntime::BufferRegion& region,
                                   Body body)
            {
                subject.port().kernel(std::size_t{1},
                                      SushiRuntime::Access(SushiRuntime::Reads(),
                                                           SushiRuntime::Writes(region)),
                                      body);
            }

            /** @brief Records a node that reads and writes region and runs body once. */
            template <typename Subject, typename Body>
            static void update_once(Subject& subject, const SushiRuntime::BufferRegion& region,
                                    Body body)
            {
                subject.port().kernel(std::size_t{1},
                                      SushiRuntime::Access(SushiRuntime::Reads(region),
                                                           SushiRuntime::Writes(region)),
                                      body);
            }

            /** @brief Nodes recorded one after the other run in the order they were recorded. */
            template <typename Subject>
            static void expect_record_order_is_run_order(Subject& subject)
            {
                std::uint32_t* out = subject.output();
                const SushiRuntime::BufferRegion region = region_of(out, 1);
                write_once(subject, region, [out](std::size_t) { out[0] = 1u; });
                update_once(subject, region, [out](std::size_t) { out[0] = out[0] * 10u + 2u; });
                update_once(subject, region, [out](std::size_t) { out[0] = out[0] * 10u + 3u; });

                subject.run();

                EXPECT_EQ(out[0], 123u);
            }

            /** @brief A scratch block is written by one node and read by the next. */
            template <typename Subject>
            static void expect_scratch_carries_a_value_between_nodes(Subject& subject)
            {
                std::uint32_t* out = subject.output();
                auto block = subject.port().template scratch<std::uint32_t>(
                    1, subject.device(), subject.residency());
                ASSERT_TRUE(block);
                std::uint32_t* scratch = *block;
                const SushiRuntime::BufferRegion scratch_region = region_of(scratch, 1);
                write_once(subject, scratch_region, [scratch](std::size_t) { scratch[0] = 41u; });
                subject.port().kernel(
                    std::size_t{1},
                    SushiRuntime::Access(SushiRuntime::Reads(scratch_region),
                                         SushiRuntime::Writes(region_of(out, 1))),
                    [scratch, out](std::size_t) { out[0] = scratch[0] + 1u; });

                subject.run();

                EXPECT_EQ(out[0], 42u);
            }

            /** @brief A resolve hands the live count of the run to its fill. */
            template <typename Subject>
            static void expect_resolve_hands_over_the_live_count(Subject& subject)
            {
                std::uint32_t* out = subject.output();
                const SushiRuntime::LateCount count = SushiRuntime::LateCount::from_host(
                    K_CONTRACT_ITEMS, []() { return K_CONTRACT_LIVE_COUNT; });
                subject.port().resolve(
                    std::array<SushiRuntime::LateCount, 1>{count}, subject.device(),
                    SushiRuntime::Access(SushiRuntime::Reads(),
                                         SushiRuntime::Writes(region_of(out, 1))),
                    [out](const std::array<std::size_t, 1>& live)
                    {
                        out[0] = static_cast<std::uint32_t>(live[0]);
                    });

                subject.run();

                EXPECT_EQ(out[0], static_cast<std::uint32_t>(K_CONTRACT_LIVE_COUNT));
            }

            /** @brief A device-sized launch runs only the items below its live width. */
            template <typename Subject>
            static void expect_a_device_sized_launch_stops_at_its_live_width(Subject& subject)
            {
                std::uint32_t* out = subject.output();
                auto width_block = subject.port().template scratch<std::uint32_t>(
                    1, subject.device(), subject.residency());
                ASSERT_TRUE(width_block);
                std::uint32_t* width = *width_block;
                const SushiRuntime::BufferRegion width_region = region_of(width, 1);
                write_once(subject, width_region,
                           [width](std::size_t) { width[0] = K_CONTRACT_LIVE_WIDTH; });
                subject.port().kernel(
                    SushiRuntime::DeviceSized{K_CONTRACT_ITEMS, width, nullptr},
                    SushiRuntime::Access(SushiRuntime::Reads(width_region),
                                         SushiRuntime::Writes(region_of(out, K_CONTRACT_ITEMS))),
                    [out](std::size_t t) { out[t] = 1u; });

                subject.run();

                std::size_t ran = 0;
                for (std::size_t t = 0; t < K_CONTRACT_ITEMS; ++t)
                    ran += out[t];
                EXPECT_EQ(ran, std::size_t{K_CONTRACT_LIVE_WIDTH});
            }

            /** @brief Nodes recorded inside one rule scope belong to one rule. */
            template <typename Subject>
            static void expect_a_rule_scope_groups_its_nodes(Subject& subject)
            {
                std::uint32_t* out = subject.output();
                const SushiRuntime::BufferRegion region = region_of(out, 1);
                {
                    auto rule = subject.port().rule();
                    write_once(subject, region, [out](std::size_t) { out[0] = 1u; });
                    update_once(subject, region, [out](std::size_t) { out[0] = 2u; });
                }

                EXPECT_EQ(subject.rule_count(), std::size_t{1});
            }

            /** @brief A recording nothing refused reports no refusal. */
            template <typename Subject>
            static void expect_a_clean_recording_reports_no_refusal(Subject& subject)
            {
                std::uint32_t* out = subject.output();
                write_once(subject, region_of(out, 1), [out](std::size_t) { out[0] = 1u; });

                EXPECT_EQ(subject.port().refusal(), nullptr);
            }
        };
    } // namespace Test
} // namespace SushiBLAS
