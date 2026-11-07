/**************************************************************************/
/* test_host_primitive_recorder.cpp                                       */
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
 * @file test_host_primitive_recorder.cpp
 * @brief Tests that HostPrimitiveRecorder keeps the PrimitiveRecorder contract and the record it keeps of each node.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

#include "support/primitive_recorder_contract.hpp"
#include "support/primitive_recorder_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the PrimitiveRecorder contract against a subject on the host. */
        template <typename Subject>
        class Small_PrimitiveRecorderContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostPrimitiveSubjects = ::testing::Types<HostPrimitiveSubject>;
        SB_TYPED_TEST_SUITE(Small_PrimitiveRecorderContract, HostPrimitiveSubjects);

        TYPED_TEST(Small_PrimitiveRecorderContract, RecordOrderIsRunOrder)
        {
            PrimitiveRecorderContract::expect_record_order_is_run_order(this->subject);
        }

        TYPED_TEST(Small_PrimitiveRecorderContract, ScratchCarriesAValueBetweenNodes)
        {
            PrimitiveRecorderContract::expect_scratch_carries_a_value_between_nodes(this->subject);
        }

        TYPED_TEST(Small_PrimitiveRecorderContract, ResolveHandsOverTheLiveCount)
        {
            PrimitiveRecorderContract::expect_resolve_hands_over_the_live_count(this->subject);
        }

        TYPED_TEST(Small_PrimitiveRecorderContract, ADeviceSizedLaunchStopsAtItsLiveWidth)
        {
            PrimitiveRecorderContract::expect_a_device_sized_launch_stops_at_its_live_width(this->subject);
        }

        TYPED_TEST(Small_PrimitiveRecorderContract, ARuleScopeGroupsItsNodes)
        {
            PrimitiveRecorderContract::expect_a_rule_scope_groups_its_nodes(this->subject);
        }

        TYPED_TEST(Small_PrimitiveRecorderContract, ACleanRecordingReportsNoRefusal)
        {
            PrimitiveRecorderContract::expect_a_clean_recording_reports_no_refusal(this->subject);
        }

        /** @brief Tests that a node keeps the name, class and width its options state. */
        TEST(Small_HostPrimitiveRecorder, KeepsTheOptionsOfEachNode)
        {
            HostPrimitiveSubject subject;
            SushiRuntime::NodeOptions options;
            options.name = "stage";
            options.determinism = SushiRuntime::Determinism::Bitwise;
            options.width = 4;

            subject.port().kernel(
                std::size_t{4},
                SushiRuntime::Access(SushiRuntime::Reads(),
                                     SushiRuntime::Writes(PrimitiveRecorderContract::region_of(
                                         subject.output(), 4))),
                [](std::size_t) {}, options);

            ASSERT_EQ(subject.port().nodes().size(), std::size_t{1});
            const GraphPrimitives::PrimitiveNodeRecord& node = subject.port().nodes().front();
            EXPECT_EQ(node.name, "stage");
            EXPECT_EQ(node.determinism, SushiRuntime::Determinism::Bitwise);
            EXPECT_EQ(node.width, std::size_t{4});
            EXPECT_EQ(node.kind, GraphPrimitives::PrimitiveNodeKind::Kernel);
        }

        /** @brief Tests that a resolve records the capacity of each late count it resolves. */
        TEST(Small_HostPrimitiveRecorder, ARecordedResolveKeepsTheCapacityOfItsCounts)
        {
            HostPrimitiveSubject subject;
            const SushiRuntime::LateCount count = SushiRuntime::LateCount::fixed(K_CONTRACT_ITEMS);

            subject.port().resolve(
                std::array<SushiRuntime::LateCount, 1>{count}, subject.device(),
                SushiRuntime::Access(), [](const std::array<std::size_t, 1>&) {});

            ASSERT_EQ(subject.port().nodes().size(), std::size_t{1});
            EXPECT_EQ(subject.port().nodes().front().kind, GraphPrimitives::PrimitiveNodeKind::Resolve);
            EXPECT_EQ(subject.port().nodes().front().late_capacities,
                      std::vector<std::size_t>{K_CONTRACT_ITEMS});
        }

        /** @brief Tests that a run fails when a live count exceeds its capacity. */
        TEST(Small_HostPrimitiveRecorder, ALiveCountOverItsCapacityFailsTheRun)
        {
            HostPrimitiveSubject subject;
            const SushiRuntime::LateCount count = SushiRuntime::LateCount::from_host(
                K_CONTRACT_ITEMS, []() { return K_CONTRACT_ITEMS + 1; });
            subject.port().resolve(
                std::array<SushiRuntime::LateCount, 1>{count}, subject.device(),
                SushiRuntime::Access(), [](const std::array<std::size_t, 1>&) {});

            const SushiRuntime::Result<void> ran = subject.port().run();

            ASSERT_FALSE(ran);
            EXPECT_EQ(ran.error().code, SushiRuntime::ErrorCode::KernelFailed);
        }

        /** @brief Tests that nodes_touching counts the readers and the writers of a region. */
        TEST(Small_HostPrimitiveRecorder, NodesTouchingCountsReadersAndWritersOfARegion)
        {
            HostPrimitiveSubject subject;
            const SushiRuntime::BufferRegion first =
                PrimitiveRecorderContract::region_of(subject.output(), 2);
            const SushiRuntime::BufferRegion second =
                PrimitiveRecorderContract::region_of(subject.output() + 2, 2);
            subject.port().kernel(std::size_t{1},
                                  SushiRuntime::Access(SushiRuntime::Reads(),
                                                       SushiRuntime::Writes(first)),
                                  [](std::size_t) {});
            subject.port().kernel(std::size_t{1},
                                  SushiRuntime::Access(SushiRuntime::Reads(first),
                                                       SushiRuntime::Writes(second)),
                                  [](std::size_t) {});

            EXPECT_EQ(subject.port().nodes_touching(first), std::size_t{2});
            EXPECT_EQ(subject.port().nodes_touching(second), std::size_t{1});
            EXPECT_EQ(subject.port().nodes_touching(
                          PrimitiveRecorderContract::region_of(subject.output() + 4, 2)),
                      std::size_t{0});
        }

    } // namespace Test
} // namespace SushiBLAS
