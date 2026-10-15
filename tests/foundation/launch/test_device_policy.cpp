/**************************************************************************/
/* test_device_policy.cpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_device_policy.cpp
 * @brief Tests the cache blocking, the group sizes, the 1D grid and the vector width of a profile.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a GPU profile with the given unit count, group limit and sub-group width.
        Device::Profile gpu_profile(std::size_t units, std::size_t max_group, std::size_t width)
        {
            Device::Profile profile;
            profile.is_cpu = false;
            profile.compute_units = units;
            profile.max_work_group_size = max_group;
            profile.sub_group_size = width;
            return profile;
        }

        /// @brief Returns a CPU profile with the given unit count and group limit.
        Device::Profile cpu_profile(std::size_t units, std::size_t max_group)
        {
            Device::Profile profile;
            profile.is_cpu = true;
            profile.compute_units = units;
            profile.max_work_group_size = max_group;
            profile.sub_group_size = 1;
            return profile;
        }

        TEST(Unit_DevicePolicy, CacheBlockingScalesWithTheCache)
        {
            const Device::CacheBlocking blocking = Device::cache_blocking_for(1024 * 1024);

            EXPECT_EQ(blocking.outer_bytes, std::size_t{256 * 1024});
            EXPECT_EQ(blocking.stream_bytes, std::size_t{512 * 1024});
            EXPECT_EQ(blocking.min_operand_bytes, std::size_t{1024 * 1024});
            EXPECT_EQ(blocking.inner_bytes, Device::CacheBlocking{}.inner_bytes);
        }

        TEST(Unit_DevicePolicy, TheDefaultBlockingIsTheBlockingOfTheAssumedCache)
        {
            const Device::CacheBlocking assumed =
                Device::cache_blocking_for(Device::K_ASSUMED_CACHE_BYTES);
            const Device::CacheBlocking fallback{};

            EXPECT_EQ(assumed.outer_bytes, fallback.outer_bytes);
            EXPECT_EQ(assumed.stream_bytes, fallback.stream_bytes);
            EXPECT_EQ(assumed.min_operand_bytes, fallback.min_operand_bytes);
        }

        TEST(Unit_DevicePolicy, AGPUGroupIsAWholeNumberOfSubGroupsUnderItsTarget)
        {
            EXPECT_EQ(Device::work_group_size(gpu_profile(10, 1024, 32)), std::size_t{256});
            EXPECT_EQ(Device::work_group_size(gpu_profile(10, 100, 32)), std::size_t{96});
            EXPECT_EQ(Device::work_group_size(gpu_profile(10, 16, 32)), std::size_t{1});
        }

        TEST(Unit_DevicePolicy, ACPUGroupGivesEachUnitItsShareAboveTheFloor)
        {
            const Device::Profile cpu = cpu_profile(4, 8192);

            EXPECT_EQ(Device::flat_work_group_size(cpu, 100), Device::K_CPU_WORK_GROUP_TARGET);
            EXPECT_EQ(Device::flat_work_group_size(cpu, 4000), std::size_t{1000});
            EXPECT_EQ(Device::flat_work_group_size(cpu, 1 << 20), std::size_t{8192});
        }

        TEST(Unit_DevicePolicy, AGPUFlatGroupIgnoresTheElementCount)
        {
            const Device::Profile gpu = gpu_profile(10, 1024, 32);

            EXPECT_EQ(Device::flat_work_group_size(gpu, 7), Device::work_group_size(gpu));
            EXPECT_EQ(Device::flat_work_group_size(gpu, 1 << 24), Device::work_group_size(gpu));
        }

        TEST(Unit_DevicePolicy, AGridCoversTheElementsInWholeGroups)
        {
            const Device::Grid1D grid = Device::plan_1d(cpu_profile(4, 8192), 4000);

            EXPECT_EQ(grid.local_size, std::size_t{1000});
            EXPECT_EQ(grid.global_size % grid.local_size, std::size_t{0});
            EXPECT_GE(grid.global_size, std::size_t{4000});
            EXPECT_EQ(grid.groups(), grid.global_size / grid.local_size);
            EXPECT_EQ(grid.stride(), grid.global_size);
        }

        TEST(Unit_DevicePolicy, AGPUGridStopsAtTheResidentGroups)
        {
            const Device::Profile gpu = gpu_profile(2, 1024, 32);
            const Device::Grid1D grid = Device::plan_1d(gpu, std::size_t{1} << 24);

            EXPECT_EQ(grid.groups(), 2 * Device::K_WAVES_PER_COMPUTE_UNIT);
        }

        TEST(Unit_DevicePolicy, UnrollingDividesTheGroupsAGridNeeds)
        {
            const Device::Profile cpu = cpu_profile(1, 64);

            EXPECT_EQ(Device::plan_1d(cpu, 640, 1).groups(), std::size_t{10});
            EXPECT_EQ(Device::plan_1d(cpu, 640, 4).groups(), std::size_t{3});
            EXPECT_EQ(Device::plan_1d(cpu, 0).groups(), std::size_t{1});
        }

        TEST(Unit_DevicePolicy, TheGridNdRangeMatchesItsSizes)
        {
            const Device::Grid1D grid{512, 64};
            const sycl::nd_range<1> range = grid.nd_range();

            EXPECT_EQ(range.get_global_range()[0], std::size_t{512});
            EXPECT_EQ(range.get_local_range()[0], std::size_t{64});
        }

        TEST(Unit_DevicePolicy, VectorWidthFillsOneTransactionOnAGPUOnly)
        {
            const Device::Profile gpu = gpu_profile(10, 1024, 32);

            EXPECT_EQ(Device::vector_width(gpu, 4), std::size_t{4});
            EXPECT_EQ(Device::vector_width(gpu, 2), std::size_t{8});
            EXPECT_EQ(Device::vector_width(gpu, 16), std::size_t{1});
            EXPECT_EQ(Device::vector_width(gpu, 0), std::size_t{1});
            EXPECT_EQ(Device::vector_width(cpu_profile(4, 64), 4), std::size_t{1});
        }

        TEST(Unit_DevicePolicy, ALocalTileFitsWhenMemoryAndGroupBothSuffice)
        {
            Device::Profile profile = gpu_profile(10, 256, 32);
            profile.local_mem_bytes = 48 * 1024;

            EXPECT_TRUE(Device::hosts_local_tile(profile, {48 * 1024, 256}));
            EXPECT_FALSE(Device::hosts_local_tile(profile, {48 * 1024 + 1, 256}));
            EXPECT_FALSE(Device::hosts_local_tile(profile, {1024, 257}));
        }

        TEST(Unit_DevicePolicy, ProfileForProbesADeviceOnceAndKeepsWhatItSaw)
        {
            const sycl::queue& queue = TestEnvironment::queue(0);
            const Device::Profile& first = Device::profile_for(queue);
            const Device::Profile& second = Device::profile_for(queue.get_device());

            EXPECT_EQ(&first, &second);
            EXPECT_EQ(first.is_cpu, queue.get_device().is_cpu());
            EXPECT_GE(first.compute_units, std::size_t{1});
            EXPECT_GE(first.max_work_group_size, std::size_t{1});
            EXPECT_GE(first.sub_group_size, std::size_t{1});
            EXPECT_EQ(first.blocking.min_operand_bytes, first.cache_bytes);
        }
    } // namespace Test
} // namespace SushiBLAS
