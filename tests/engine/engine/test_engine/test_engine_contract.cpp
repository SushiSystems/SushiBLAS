/**************************************************************************/
/* test_engine_contract.cpp                                               */
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
 * @file test_engine_contract.cpp
 * @brief Tests that a host engine and an engine on a runtime keep the Engine contract, and what only a host engine refuses.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/engine_contract.hpp"
#include "support/engine_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the Engine contract against a subject on the host. */
        template<typename Subject>
        class Small_EngineContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Runs the Engine contract against a subject on the shared runtime. */
        template<typename Subject>
        class Medium_EngineContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostEngineSubjects = ::testing::Types<HostEngineSubject>;
        using RuntimeEngineSubjects = ::testing::Types<RuntimeEngineSubject>;
        SB_TYPED_TEST_SUITE(Small_EngineContract, HostEngineSubjects);
        SB_TYPED_TEST_SUITE(Medium_EngineContract, RuntimeEngineSubjects);

        TYPED_TEST(Small_EngineContract, ATensorHoldsTheBytesWrittenToIt)
        {
            EngineContract::expect_tensor_bytes_read_back(this->subject);
        }

        TYPED_TEST(Small_EngineContract, AReferenceTaskRecordsAndRuns)
        {
            EngineContract::expect_reference_task_records_and_runs(this->subject);
        }

        TYPED_TEST(Small_EngineContract, AScratchLeaseReturnsItsBlock)
        {
            EngineContract::expect_scratch_lease_and_return(this->subject);
        }

        TYPED_TEST(Small_EngineContract, ForceReferenceIsAccepted)
        {
            EngineContract::expect_force_reference_is_accepted(this->subject);
        }

        TYPED_TEST(Small_EngineContract, OtherModesFollowTheRuntime)
        {
            EngineContract::expect_other_modes_follow_the_runtime(this->subject);
        }

        TYPED_TEST(Small_EngineContract, TheRuntimeFollowsTheKind)
        {
            EngineContract::expect_runtime_follows_the_kind(this->subject);
        }

        TYPED_TEST(Small_EngineContract, TheRandomStreamReservesOffsets)
        {
            EngineContract::expect_random_stream_reserves_offsets(this->subject);
        }

        TYPED_TEST(Medium_EngineContract, ATensorHoldsTheBytesWrittenToIt)
        {
            EngineContract::expect_tensor_bytes_read_back(this->subject);
        }

        TYPED_TEST(Medium_EngineContract, AReferenceTaskRecordsAndRuns)
        {
            EngineContract::expect_reference_task_records_and_runs(this->subject);
        }

        TYPED_TEST(Medium_EngineContract, AScratchLeaseReturnsItsBlock)
        {
            EngineContract::expect_scratch_lease_and_return(this->subject);
        }

        TYPED_TEST(Medium_EngineContract, ForceReferenceIsAccepted)
        {
            EngineContract::expect_force_reference_is_accepted(this->subject);
        }

        TYPED_TEST(Medium_EngineContract, OtherModesFollowTheRuntime)
        {
            EngineContract::expect_other_modes_follow_the_runtime(this->subject);
        }

        TYPED_TEST(Medium_EngineContract, TheRuntimeFollowsTheKind)
        {
            EngineContract::expect_runtime_follows_the_kind(this->subject);
        }

        TYPED_TEST(Medium_EngineContract, TheRandomStreamReservesOffsets)
        {
            EngineContract::expect_random_stream_reserves_offsets(this->subject);
        }

        /** @brief Tests that a host engine starts in ForceReference, the one mode it runs. */
        TEST(Small_HostEngine, ItStartsInForceReference)
        {
            const Engine engine(Engine::Host{});

            EXPECT_EQ(engine.mode(), ExecutionMode::ForceReference);
        }

        /** @brief Tests that a host engine creates host tensors in the layout it was built with. */
        TEST(Small_HostEngine, ItCreatesHostTensorsInItsLayout)
        {
            Engine engine(Engine::Host{}, Core::Layout::ColumnMajor);

            const Tensor tensor = engine.create_tensor({2, 3});

            EXPECT_TRUE(tensor.storage->is_host());
            EXPECT_EQ(engine.layout(), Core::Layout::ColumnMajor);
            EXPECT_EQ(tensor.layout, Core::Layout::ColumnMajor);
        }

        /** @brief Tests that a host engine has the sink its recorders record through, and a runtime engine does not. */
        TEST(Small_HostEngine, ItOwnsTheSinkItsRecordersUse)
        {
            Engine engine(Engine::Host{});

            EXPECT_EQ(engine.host_sink().recorded_count(), std::size_t{0});
        }

        /** @brief Tests that an engine on a runtime has no host sink. */
        TEST(Medium_RuntimeEngine, ItHasNoHostSink)
        {
            Engine engine(TestEnvironment::runtime());

            EXPECT_THROW((void)engine.host_sink(), UnsupportedExecution);
        }
    } // namespace Test
} // namespace SushiBLAS
