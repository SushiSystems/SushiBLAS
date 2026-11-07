/**************************************************************************/
/* test_runtime_scratch_source.cpp                                        */
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
 * @file test_runtime_scratch_source.cpp
 * @brief Tests what RuntimeScratchSource serves beyond the BlockSource contract.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that the source serves device memory of the runtime it was built on. */
        TEST(Medium_RuntimeScratchSource, ItServesDeviceMemoryOfItsRuntime)
        {
            RuntimeScratchSource source(TestEnvironment::runtime());

            const auto block = source.serve(256, 0);

            EXPECT_EQ(block->residency(), SushiRuntime::Residency::Device);
            EXPECT_FALSE(block->is_host());
            EXPECT_EQ(&source.runtime(), &TestEnvironment::runtime());
        }

        /** @brief Tests that the source reports the device count of the runtime. */
        TEST(Medium_RuntimeScratchSource, ItReportsTheDeviceCountOfItsRuntime)
        {
            const RuntimeScratchSource source(TestEnvironment::runtime());

            EXPECT_EQ(source.device_count(), TestEnvironment::device_count());
        }
    } // namespace Test
} // namespace SushiBLAS
