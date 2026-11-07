/**************************************************************************/
/* test_block_maker.cpp                                                   */
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
 * @file test_block_maker.cpp
 * @brief Tests that the host and runtime adapters of BlockMaker size, place and report their blocks.
 * @author Mustafa Garip
 */

#include <memory>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/tensor/block_maker.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_HostBlockMaker, ItHasOneDevice)
        {
            const HostBlockMaker maker;

            EXPECT_EQ(maker.device_count(), std::size_t{1});
        }

        TEST(Small_HostBlockMaker, ItMakesHostMemoryWhateverResidencyIsAsked)
        {
            const HostBlockMaker maker;

            const std::unique_ptr<StorageBlock> block =
                maker.make(10, SushiRuntime::Residency::Device, 0);

            EXPECT_EQ(block->residency(), SushiRuntime::Residency::Host);
        }

        TEST(Small_HostBlockMaker, ItRoundsTheSizeUpToTheAlignment)
        {
            const HostBlockMaker maker;

            const std::unique_ptr<StorageBlock> block =
                maker.make(10, SushiRuntime::Residency::Host, 0);

            EXPECT_EQ(block->size_bytes(), K_STORAGE_ALIGNMENT);
        }

        TEST(Medium_RuntimeBlockMaker, ItKeepsItsRuntime)
        {
            const RuntimeBlockMaker maker(TestEnvironment::runtime());

            EXPECT_EQ(&maker.runtime(), &TestEnvironment::runtime());
        }

        TEST(Medium_RuntimeBlockMaker, ItCountsTheDevicesOfItsRuntime)
        {
            const RuntimeBlockMaker maker(TestEnvironment::runtime());

            EXPECT_EQ(maker.device_count(), TestEnvironment::device_count());
        }

        TEST(Medium_RuntimeBlockMaker, ItPlacesABlockInTheResidencyItIsAsked)
        {
            const RuntimeBlockMaker maker(TestEnvironment::runtime());

            const std::unique_ptr<StorageBlock> block =
                maker.make(10, SushiRuntime::Residency::Device, 0);

            EXPECT_EQ(block->residency(), SushiRuntime::Residency::Device);
        }

        TEST(Medium_RuntimeBlockMaker, ItRefusesADeviceTheRuntimeDoesNotDrive)
        {
            const RuntimeBlockMaker maker(TestEnvironment::runtime());

            EXPECT_THROW((void)maker.make(10, SushiRuntime::Residency::Shared,
                                          TestEnvironment::device_count()),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
