/**************************************************************************/
/* test_runtime_primitive_recorder.cpp                                    */
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
 * @file test_runtime_primitive_recorder.cpp
 * @brief Tests that RuntimePrimitiveRecorder keeps the PrimitiveRecorder contract on a runtime graph.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <gtest/gtest.h>

#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

#include "support/primitive_recorder_contract.hpp"
#include "support/primitive_recorder_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the PrimitiveRecorder contract against a subject on a runtime graph. */
        template <typename Subject>
        class Medium_PrimitiveRecorderContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using RuntimePrimitiveSubjects = ::testing::Types<RuntimePrimitiveSubject>;
        SB_TYPED_TEST_SUITE(Medium_PrimitiveRecorderContract, RuntimePrimitiveSubjects);

        TYPED_TEST(Medium_PrimitiveRecorderContract, RecordOrderIsRunOrder)
        {
            PrimitiveRecorderContract::expect_record_order_is_run_order(this->subject);
        }

        TYPED_TEST(Medium_PrimitiveRecorderContract, ScratchCarriesAValueBetweenNodes)
        {
            PrimitiveRecorderContract::expect_scratch_carries_a_value_between_nodes(this->subject);
        }

        TYPED_TEST(Medium_PrimitiveRecorderContract, ResolveHandsOverTheLiveCount)
        {
            PrimitiveRecorderContract::expect_resolve_hands_over_the_live_count(this->subject);
        }

        TYPED_TEST(Medium_PrimitiveRecorderContract, ADeviceSizedLaunchStopsAtItsLiveWidth)
        {
            PrimitiveRecorderContract::expect_a_device_sized_launch_stops_at_its_live_width(this->subject);
        }

        TYPED_TEST(Medium_PrimitiveRecorderContract, ARuleScopeGroupsItsNodes)
        {
            PrimitiveRecorderContract::expect_a_rule_scope_groups_its_nodes(this->subject);
        }

        TYPED_TEST(Medium_PrimitiveRecorderContract, ACleanRecordingReportsNoRefusal)
        {
            PrimitiveRecorderContract::expect_a_clean_recording_reports_no_refusal(this->subject);
        }
    } // namespace Test
} // namespace SushiBLAS
