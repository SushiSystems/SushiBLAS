/**************************************************************************/
/* test_host_engine_test.cpp                                              */
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
 * @file test_host_engine_test.cpp
 * @brief Tests that HostEngineTest gives a test a host engine, a recorder that runs on the host and read-back by memcpy.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Gives each test the host engine and recorder of HostEngineTest. */
        class Small_HostEngineTest : public HostEngineTest
        {
        };

        TEST_F(Small_HostEngineTest, TheEngineIsAHostEngineInForceReference)
        {
            EXPECT_TRUE(engine->is_host());
            EXPECT_EQ(engine->mode(), ExecutionMode::ForceReference);
            EXPECT_EQ(recorder().mode(), ExecutionMode::ForceReference);
        }

        TEST_F(Small_HostEngineTest, AnInitializedTensorHoldsItsValues)
        {
            const Tensor tensor =
                create_initialized_tensor<float>({2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

            EXPECT_EQ(read_tensor<float>(tensor), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
        }

        TEST_F(Small_HostEngineTest, ReinitEngineChangesTheLayoutAndKeepsTheRecorderOnTheHost)
        {
            reinit_engine(Core::Layout::ColumnMajor);

            EXPECT_EQ(engine->layout(), Core::Layout::ColumnMajor);
            EXPECT_TRUE(engine->is_host());
            EXPECT_EQ(recorder().mode(), ExecutionMode::ForceReference);
        }

        TEST_F(Small_HostEngineTest, ExpectTensorReadsAHostTensorWithoutAQueue)
        {
            const Tensor tensor =
                create_initialized_tensor<float>({3}, std::vector<float>{0.5f, -1.0f, 2.0f});

            expect_tensor_equal(tensor, {0.5f, -1.0f, 2.0f});
        }
    } // namespace Test
} // namespace SushiBLAS
