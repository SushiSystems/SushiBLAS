/**************************************************************************/
/* test_recording.cpp                                                     */
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

/**
 * @file test_recording.cpp
 * @brief Tests what IndexOps::for_each records and refuses, on a host engine with no device.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/index/index_ops.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /// @brief The capacity of the count-refusal rows.
            constexpr std::size_t K_CAPACITY = 4;

            /// @brief The operands an update declares twice.
            enum class Overlap
            {
                ReadsWhatItUpdates,         ///< A read of the tensor the task updates.
                WritesWhatItUpdates,        ///< A write of the tensor the task updates.
                WritesAViewOfWhatItUpdates, ///< A write of a view into the updated tensor.
                UpdatesAViewOfWhatItUpdates ///< A second update of a view into the tensor.
            };

            /// @brief One overlapping declaration and the count the task is recorded with.
            struct OverlapRow
            {
                const char* name;
                Overlap overlap;
                std::size_t count;
            };

            /// @brief Returns the overlapping declarations, and one at a count of zero.
            std::vector<OverlapRow> overlap_rows()
            {
                return {{"ReadOfTheUpdatedTensor", Overlap::ReadsWhatItUpdates, 4},
                        {"WriteOfTheUpdatedTensor", Overlap::WritesWhatItUpdates, 4},
                        {"WriteOfAViewOfTheUpdatedTensor", Overlap::WritesAViewOfWhatItUpdates, 2},
                        {"UpdateOfAViewOfTheUpdatedTensor", Overlap::UpdatesAViewOfWhatItUpdates, 2},
                        {"ReadOfTheUpdatedTensorAtACountOfZero", Overlap::ReadsWhatItUpdates, 0}};
            }

            /// @brief One count tensor for_each refuses: what it holds and the capacity it sizes.
            struct CountRefusalRow
            {
                const char* name;
                Core::DataType type;
                std::size_t capacity;
            };

            /// @brief Returns the refused count tensors of a host engine.
            std::vector<CountRefusalRow> count_refusal_rows()
            {
                return {{"Float", Core::DataType::Float32, K_CAPACITY},
                        {"Int64", Core::DataType::Int64, K_CAPACITY},
                        {"FloatAtACapacityOfZero", Core::DataType::Float32, 0}};
            }

            /// @brief Records one for_each whose operands overlap as the row says.
            void record_overlap(IndexOps& index, const OverlapRow& row, Tensor& x, Tensor& head)
            {
                const OperationOptions options{"overlap", SushiRuntime::Determinism::Bitwise};
                switch (row.overlap)
                {
                    case Overlap::ReadsWhatItUpdates:
                        index.for_each(row.count,
                                       Dispatch::bind_updates(
                                           Dispatch::bind_reads(
                                               [](std::size_t, const float*, float*) {},
                                               Dispatch::read_as<float>(x)),
                                           Dispatch::update_as<float>(x)),
                                       options);
                        break;
                    case Overlap::WritesWhatItUpdates:
                        index.for_each(row.count,
                                       Dispatch::bind_writes(
                                           Dispatch::bind_updates(
                                               [](std::size_t, float*, float*) {},
                                               Dispatch::update_as<float>(x)),
                                           Dispatch::write_as<float>(x)),
                                       options);
                        break;
                    case Overlap::WritesAViewOfWhatItUpdates:
                        index.for_each(row.count,
                                       Dispatch::bind_writes(
                                           Dispatch::bind_updates(
                                               [](std::size_t, float*, float*) {},
                                               Dispatch::update_as<float>(x)),
                                           Dispatch::write_as<float>(head)),
                                       options);
                        break;
                    case Overlap::UpdatesAViewOfWhatItUpdates:
                        index.for_each(row.count,
                                       Dispatch::bind_updates(
                                           [](std::size_t, float*, float*) {},
                                           Dispatch::update_as<float>(x),
                                           Dispatch::update_as<float>(head)),
                                       options);
                        break;
                }
            }
        } // namespace

        /** @brief Gives each test a host engine and the per-index view over its recorder. */
        class Small_IndexOps : public HostEngineTest
        {
        };

        /** @brief Tests that a fixed count of zero records nothing and returns no handle. */
        TEST_F(Small_IndexOps, ACountOfZeroRecordsNothing)
        {
            IndexOps index(recorder());
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);

            const TaskHandle handle = index.for_each(
                0, Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                         Dispatch::write_as<float>(out)),
                OperationOptions{"empty", SushiRuntime::Determinism::Bitwise});

            EXPECT_FALSE(handle.is_valid());
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Tests that the recorded task carries the name and the operation id. */
        TEST_F(Small_IndexOps, TheNodeCarriesTheNameOfItsOptionsAndTheOperationId)
        {
            IndexOps index(recorder());
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);

            index.for_each(4,
                           Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                 Dispatch::write_as<float>(out)),
                           OperationOptions{"fill_out", SushiRuntime::Determinism::Tolerant});

            const std::vector<HostTaskRecord> records = recording->records();
            ASSERT_EQ(records.size(), std::size_t{1});
            EXPECT_EQ(records[0].name, "fill_out");
            EXPECT_EQ(records[0].operation, "index.for_each"_op);
        }

        /** @brief Tests that the determinism class of the options reaches the recorded task. */
        TEST_F(Small_IndexOps, TheNodeCarriesTheDeterminismClassOfItsOptions)
        {
            IndexOps index(recorder());
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);

            index.for_each(4,
                           Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                 Dispatch::write_as<float>(out)),
                           OperationOptions{"fill_out", SushiRuntime::Determinism::Cosmetic});

            const std::vector<HostTaskRecord> records = recording->records();
            ASSERT_EQ(records.size(), std::size_t{1});
            EXPECT_EQ(records[0].determinism, SushiRuntime::Determinism::Cosmetic);
        }

        /** @brief Tests that an update declares one read and one write. */
        TEST_F(Small_IndexOps, AnUpdateDeclaresOneReadAndOneWrite)
        {
            IndexOps index(recorder());
            Tensor x = engine->create_tensor({4}, Core::DataType::Float32);

            index.for_each(4,
                           Dispatch::bind_updates([](std::size_t i, float* p) { p[i] += 1.0f; },
                                                  Dispatch::update_as<float>(x)),
                           OperationOptions{"bump_x", SushiRuntime::Determinism::Bitwise});

            const std::vector<HostTaskRecord> records = recording->records();
            ASSERT_EQ(records.size(), std::size_t{1});
            EXPECT_EQ(records[0].reads.size(), std::size_t{1});
            EXPECT_EQ(records[0].writes.size(), std::size_t{1});
        }

        /** @brief Tests that a task is refused when it writes the count that sizes it. */
        TEST_F(Small_IndexOps, ATaskThatWritesItsOwnCountIsRefused)
        {
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32);
            IndexOps index(recorder());

            EXPECT_THROW(index.for_each(4, count,
                                        Dispatch::bind_writes(
                                            [](std::size_t, std::int32_t* slot) { slot[0] = 1; },
                                            Dispatch::write_as<std::int32_t>(count)),
                                        OperationOptions{"self_count",
                                                         SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Runs one count tensor of the table of count_refusal_rows(). */
        class Small_IndexOpsCountRefusal : public Small_IndexOps,
                                           public ::testing::WithParamInterface<CountRefusalRow>
        {
        };

        /** @brief Tests that a count tensor that is not INT32 or UINT32 is refused. */
        TEST_P(Small_IndexOpsCountRefusal, IsRefusedWhenItIsNotAnInt32OrUInt32)
        {
            const CountRefusalRow row = GetParam();
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine->create_tensor({1}, row.type);
            IndexOps index(recorder());

            EXPECT_THROW(index.for_each(row.capacity, count,
                                        Dispatch::bind_writes(
                                            [](std::size_t i, float* p) { p[i] = 1.0f; },
                                            Dispatch::write_as<float>(out)),
                                        OperationOptions{"bad_count",
                                                         SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_IndexOpsCountRefusal, ::testing::ValuesIn(count_refusal_rows()),
            [](const ::testing::TestParamInfo<CountRefusalRow>& info) { return info.param.name; });

        /** @brief Runs one overlapping declaration of the table of overlap_rows(). */
        class Small_IndexOpsOverlap : public Small_IndexOps,
                                      public ::testing::WithParamInterface<OverlapRow>
        {
        };

        /** @brief Tests that an update whose operands overlap is refused and records nothing. */
        TEST_P(Small_IndexOpsOverlap, IsRefusedAndRecordsNothing)
        {
            const OverlapRow row = GetParam();
            IndexOps index(recorder());
            Tensor x = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor head = x.slice(0, 0, 2);

            EXPECT_THROW(record_overlap(index, row, x, head), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_IndexOpsOverlap, ::testing::ValuesIn(overlap_rows()),
            [](const ::testing::TestParamInfo<OverlapRow>& info)
            { return std::string(info.param.name); });

    } // namespace Test
} // namespace SushiBLAS
