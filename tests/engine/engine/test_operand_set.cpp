/**************************************************************************/
/* test_operand_set.cpp                                                   */
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
 * @file test_operand_set.cpp
 * @brief Tests the regions an OperandSet declares, the storages it keeps alive and its merge.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Reads count tensors of one element each, every one over a storage of its own. */
            void read_distinct_tensors(Engine& engine, OperandSet& operands, std::size_t count)
            {
                for (std::size_t index = 0; index < count; ++index)
                    operands.reads(engine.create_tensor({2}));
            }
        } // namespace

        /** @brief Builds operand sets over tensors of a host engine. */
        class Small_OperandSet : public HostEngineTest
        {

        };

        TEST_F(Small_OperandSet, ReadsWritesAndUpdatesDeclareTheirRegions)
        {
            const Tensor a = engine->create_tensor({4});
            const Tensor b = engine->create_tensor({4});
            const Tensor c = engine->create_tensor({4});
            OperandSet operands;

            operands.reads(a).writes(b).updates(c);

            EXPECT_EQ(operands.read_regions().size(), std::size_t{2});
            EXPECT_EQ(operands.write_regions().size(), std::size_t{2});
        }

        TEST_F(Small_OperandSet, TheSetRetainsItsOperandsAndReportsNoDeviceOnlyOperand)
        {
            const Tensor a = engine->create_tensor({4});
            const Tensor b = engine->create_tensor({4});
            const Tensor c = engine->create_tensor({4});
            OperandSet operands;

            operands.reads(a).writes(b).updates(c);

            EXPECT_EQ(operands.retained().size(), std::size_t{3});
            EXPECT_FALSE(operands.has_device_only_operand());
        }

        TEST_F(Small_OperandSet, ATensorWithoutStorageDeclaresNothing)
        {
            const Tensor empty = engine->create_tensor({0});
            OperandSet operands;

            operands.reads(empty).writes(empty);

            EXPECT_TRUE(operands.read_regions().empty());
            EXPECT_TRUE(operands.write_regions().empty());
            EXPECT_EQ(operands.retained().size(), std::size_t{0});
        }

        TEST_F(Small_OperandSet, AStorageUsedBySeveralOperandsIsRetainedOnce)
        {
            const Tensor a = engine->create_tensor({8});
            OperandSet operands;

            operands.reads(a).writes(a).reads(a.slice(0, 2, 6));

            EXPECT_EQ(operands.retained().size(), std::size_t{1});
        }

        TEST_F(Small_OperandSet, MoreStoragesThanTheSixInlineSlotsAreAllRetained)
        {
            OperandSet operands;

            read_distinct_tensors(*engine, operands, 9);

            EXPECT_EQ(operands.retained().size(), std::size_t{9});
        }

        TEST_F(Small_OperandSet, DeclaredBytesCountAnUpdatedRegionTwice)
        {
            const Tensor a = engine->create_tensor({4});
            const Tensor b = engine->create_tensor({4});
            const Tensor c = engine->create_tensor({4});
            OperandSet operands;

            operands.reads(a).writes(b).updates(c);

            EXPECT_EQ(operands.declared_bytes(), std::uint64_t{4 * sizeof(float) * 4});
        }

        /** @brief Builds operand sets over tensors of the fixture's engine on the process runtime. */
        class Medium_OperandSet : public SushiBLASTest
        {

        };

        TEST_F(Medium_OperandSet, TheSetKeepsAStorageAliveAfterItsTensor)
        {
            const std::size_t before = TestEnvironment::live_allocation_count();
            OperandSet operands;
            {
                const Tensor a = engine->create_tensor({8});
                operands.reads(a);
            }

            EXPECT_EQ(TestEnvironment::live_allocation_count(), before + 1);
        }

        TEST_F(Medium_OperandSet, ADeviceTensorIsReportedAndSurvivesAMerge)
        {
            const Tensor device = engine->create_tensor({4}, Core::DataType::Float32,
                                                        SushiRuntime::Residency::Device);
            const Tensor shared = engine->create_tensor({4});
            OperandSet first;
            OperandSet second;
            first.reads(shared);
            second.writes(device);

            first.merge(second);

            EXPECT_TRUE(first.has_device_only_operand());
            EXPECT_EQ(first.read_regions().size(), std::size_t{1});
            EXPECT_EQ(first.write_regions().size(), std::size_t{1});
            EXPECT_EQ(first.retained().size(), std::size_t{2});
        }
    } // namespace Test
} // namespace SushiBLAS
