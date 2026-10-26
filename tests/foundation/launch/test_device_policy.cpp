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
#include <string>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/test_runner/log_capture.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a GPU profile with the given unit count, group limit and sub-group width.
        static Device::Profile gpu_profile(std::size_t units, std::size_t max_group,
                                           std::size_t width)
        {
            Device::Profile profile;
            profile.is_cpu = false;
            profile.compute_units = units;
            profile.max_work_group_size = max_group;
            profile.sub_group_size = width;
            return profile;
        }

        /// @brief Returns a CPU profile with the given unit count and group limit.
        static Device::Profile cpu_profile(std::size_t units, std::size_t max_group)
        {
            Device::Profile profile;
            profile.is_cpu = true;
            profile.compute_units = units;
            profile.max_work_group_size = max_group;
            profile.sub_group_size = 1;
            return profile;
        }

        TEST(Medium_DevicePolicy, CacheBlockingScalesWithTheCache)
        {
            const Device::CacheBlocking blocking = Device::cache_blocking_for(1024 * 1024);

            EXPECT_EQ(blocking.outer_bytes, std::size_t{256 * 1024});
            EXPECT_EQ(blocking.stream_bytes, std::size_t{512 * 1024});
            EXPECT_EQ(blocking.min_operand_bytes, std::size_t{1024 * 1024});
            EXPECT_EQ(blocking.inner_bytes, Device::CacheBlocking{}.inner_bytes);
        }

        TEST(Medium_DevicePolicy, TheDefaultBlockingIsTheBlockingOfTheAssumedCache)
        {
            const Device::CacheBlocking assumed =
                Device::cache_blocking_for(Device::K_ASSUMED_CACHE_BYTES);
            const Device::CacheBlocking fallback{};

            EXPECT_EQ(assumed.outer_bytes, fallback.outer_bytes);
            EXPECT_EQ(assumed.stream_bytes, fallback.stream_bytes);
            EXPECT_EQ(assumed.min_operand_bytes, fallback.min_operand_bytes);
        }

        TEST(Medium_DevicePolicy, AGPUGroupIsAWholeNumberOfSubGroupsUnderItsTarget)
        {
            EXPECT_EQ(Device::work_group_size(gpu_profile(10, 1024, 32)), std::size_t{256});
            EXPECT_EQ(Device::work_group_size(gpu_profile(10, 100, 32)), std::size_t{96});
            EXPECT_EQ(Device::work_group_size(gpu_profile(10, 16, 32)), std::size_t{1});
        }

        TEST(Medium_DevicePolicy, ACPUGroupGivesEachUnitItsShareAboveTheFloor)
        {
            const Device::Profile cpu = cpu_profile(4, 8192);

            EXPECT_EQ(Device::flat_work_group_size(cpu, 100), Device::K_CPU_WORK_GROUP_TARGET);
            EXPECT_EQ(Device::flat_work_group_size(cpu, 4000), std::size_t{1000});
            EXPECT_EQ(Device::flat_work_group_size(cpu, 1 << 20), std::size_t{8192});
        }

        TEST(Medium_DevicePolicy, AGPUFlatGroupIgnoresTheElementCount)
        {
            const Device::Profile gpu = gpu_profile(10, 1024, 32);

            EXPECT_EQ(Device::flat_work_group_size(gpu, 7), Device::work_group_size(gpu));
            EXPECT_EQ(Device::flat_work_group_size(gpu, 1 << 24), Device::work_group_size(gpu));
        }

        TEST(Medium_DevicePolicy, AGridCoversTheElementsInWholeGroups)
        {
            const Device::Grid1D grid = Device::plan_1d(cpu_profile(4, 8192), 4000);

            EXPECT_EQ(grid.local_size, std::size_t{1000});
            EXPECT_EQ(grid.global_size % grid.local_size, std::size_t{0});
            EXPECT_GE(grid.global_size, std::size_t{4000});
            EXPECT_EQ(grid.groups(), grid.global_size / grid.local_size);
            EXPECT_EQ(grid.stride(), grid.global_size);
        }

        TEST(Medium_DevicePolicy, AGPUGridStopsAtTheResidentGroups)
        {
            const Device::Profile gpu = gpu_profile(2, 1024, 32);
            const Device::Grid1D grid = Device::plan_1d(gpu, std::size_t{1} << 24);

            EXPECT_EQ(grid.groups(), 2 * Device::K_WAVES_PER_COMPUTE_UNIT);
        }

        TEST(Medium_DevicePolicy, UnrollingDividesTheGroupsAGridNeeds)
        {
            const Device::Profile cpu = cpu_profile(1, 64);

            EXPECT_EQ(Device::plan_1d(cpu, 640, 1).groups(), std::size_t{10});
            EXPECT_EQ(Device::plan_1d(cpu, 640, 4).groups(), std::size_t{3});
            EXPECT_EQ(Device::plan_1d(cpu, 0).groups(), std::size_t{1});
        }

        TEST(Medium_DevicePolicy, TheGridNdRangeMatchesItsSizes)
        {
            const Device::Grid1D grid{512, 64};
            const sycl::nd_range<1> range = grid.nd_range();

            EXPECT_EQ(range.get_global_range()[0], std::size_t{512});
            EXPECT_EQ(range.get_local_range()[0], std::size_t{64});
        }

        TEST(Medium_DevicePolicy, VectorWidthFillsOneTransactionOnAGPUOnly)
        {
            const Device::Profile gpu = gpu_profile(10, 1024, 32);

            EXPECT_EQ(Device::vector_width(gpu, 4), std::size_t{4});
            EXPECT_EQ(Device::vector_width(gpu, 2), std::size_t{8});
            EXPECT_EQ(Device::vector_width(gpu, 16), std::size_t{1});
            EXPECT_EQ(Device::vector_width(gpu, 0), std::size_t{1});
            EXPECT_EQ(Device::vector_width(cpu_profile(4, 64), 4), std::size_t{1});
        }

        TEST(Medium_DevicePolicy, ALocalTileFitsWhenMemoryAndGroupBothSuffice)
        {
            Device::Profile profile = gpu_profile(10, 256, 32);
            profile.local_mem_bytes = 48 * 1024;

            EXPECT_TRUE(Device::hosts_local_tile(profile, {48 * 1024, 256}));
            EXPECT_FALSE(Device::hosts_local_tile(profile, {48 * 1024 + 1, 256}));
            EXPECT_FALSE(Device::hosts_local_tile(profile, {1024, 257}));
        }

        TEST(Medium_DevicePolicy, ProfileForProbesADeviceOnceAndKeepsWhatItSaw)
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

        TEST(Medium_DevicePolicy, OnlyACUDADeviceOffersTheHalfMatrix)
        {
            const sycl::queue& queue = TestEnvironment::queue(0);
            const Device::Profile& profile = Device::profile_for(queue);

            if (queue.get_device().get_backend() != sycl::backend::ext_oneapi_cuda)
                EXPECT_FALSE(profile.supports_half_matrix);
            if (profile.is_cpu)
                EXPECT_FALSE(profile.supports_half_matrix);
        }

        TEST(Large_DevicePolicy, ACUDADeviceOffersTheHalfMatrixOfSixteen)
        {
            const sycl::queue& queue = TestEnvironment::queue(0);
            if (queue.get_device().get_backend() != sycl::backend::ext_oneapi_cuda)
                GTEST_SKIP() << "the device is not a CUDA device";

            // Every card from sm_70 on lists 16 x 16 x 16 for HALF x HALF -> FLOAT32.
            EXPECT_TRUE(Device::profile_for(queue).supports_half_matrix);
        }

        TEST(Small_DevicePolicy, ALocalMemoryTableEntryReplacesWhatTheDeviceReports)
        {
            EXPECT_EQ(Device::local_mem_per_compute_unit_of(Device::Architecture::NvidiaSm86,
                                                            48 * 1024),
                      std::size_t{100 * 1024});
            EXPECT_EQ(Device::local_mem_per_compute_unit_of(Device::Architecture::NvidiaSm80, 0),
                      std::size_t{164 * 1024});
            EXPECT_EQ(Device::local_mem_per_compute_unit_of(Device::Architecture::NvidiaSm90, 0),
                      std::size_t{228 * 1024});
            EXPECT_EQ(Device::local_mem_per_compute_unit_of(Device::Architecture::IntelPVC, 0),
                      std::size_t{128 * 1024});
        }

        TEST(Small_DevicePolicy, AnUnknownArchitectureKeepsTheReportedLocalMemory)
        {
            EXPECT_EQ(Device::local_mem_per_compute_unit_of(Device::Architecture::Unknown, 32768),
                      std::size_t{32768});
        }

        TEST(Small_DevicePolicy, AnArchitectureNameIsItsEnumeratorName)
        {
            EXPECT_STREQ(Device::to_string(Device::Architecture::NvidiaSm86), "NvidiaSm86");
            EXPECT_STREQ(Device::to_string(Device::Architecture::IntelPVC), "IntelPVC");
            EXPECT_STREQ(Device::to_string(Device::Architecture::Unknown), "Unknown");
        }

        TEST(Small_DevicePolicy, AProfileStartsWithoutAVendorOrAnArchitecture)
        {
            const Device::Profile profile;

            EXPECT_EQ(profile.vendor, Device::Vendor::Unknown);
            EXPECT_EQ(profile.architecture, Device::Architecture::Unknown);
            EXPECT_EQ(profile.local_mem_per_compute_unit, std::size_t{0});
        }

        TEST(Medium_DevicePolicy, EveryDeviceReportsAConsistentVendorAndArchitecture)
        {
            for (std::size_t index = 0; index < TestEnvironment::device_count(); ++index)
            {
                const sycl::queue& queue = TestEnvironment::queue(index);
                const Device::Profile& profile = Device::profile_for(queue);

                EXPECT_EQ(profile.local_mem_per_compute_unit,
                          Device::local_mem_per_compute_unit_of(profile.architecture,
                                                                profile.local_mem_bytes));
                if (profile.architecture >= Device::Architecture::NvidiaSm70 &&
                    profile.architecture <= Device::Architecture::NvidiaSm90)
                    EXPECT_EQ(profile.vendor, Device::Vendor::Nvidia);
                if (profile.is_cpu)
                    EXPECT_FALSE(profile.architecture >= Device::Architecture::NvidiaSm70 &&
                                 profile.architecture <= Device::Architecture::NvidiaSm90);
            }
        }

        TEST(Medium_DevicePolicy, ASm86CUDADeviceIsReadAsNvidiaSm86)
        {
            bool found = false;
            for (std::size_t index = 0; index < TestEnvironment::device_count(); ++index)
            {
                const sycl::queue& queue = TestEnvironment::queue(index);
                if (queue.get_device().get_backend() != sycl::backend::ext_oneapi_cuda)
                    continue;
#if defined(SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE)
                namespace Experimental = sycl::ext::oneapi::experimental;
                if (queue.get_device().get_info<Experimental::info::device::architecture>() !=
                    Experimental::architecture::nvidia_gpu_sm_86)
                    continue;

                const Device::Profile& profile = Device::profile_for(queue);
                EXPECT_EQ(profile.architecture, Device::Architecture::NvidiaSm86);
                EXPECT_EQ(profile.vendor, Device::Vendor::Nvidia);
                EXPECT_EQ(profile.local_mem_per_compute_unit, std::size_t{100 * 1024});
                found = true;
#endif
            }
            if (!found)
                GTEST_SKIP() << "no CUDA device of architecture sm_86 is present";
        }

        TEST(Small_DeviceProbeFallbacks, ADeviceListingNoSubGroupGetsWidthOneAndIsMarkedAssumed)
        {
            const Device::Detail::Resolved none = Device::Detail::resolve_sub_group_size({});
            const Device::Detail::Resolved zero = Device::Detail::resolve_sub_group_size({0});

            EXPECT_EQ(none.value, std::size_t{1});
            EXPECT_TRUE(none.assumed);
            EXPECT_EQ(zero.value, std::size_t{1});
            EXPECT_TRUE(zero.assumed);
        }

        TEST(Small_DeviceProbeFallbacks, ADeviceListingSubGroupsKeepsTheWidestAndAssumesNothing)
        {
            const Device::Detail::Resolved widest = Device::Detail::resolve_sub_group_size({16, 32, 8});

            EXPECT_EQ(widest.value, std::size_t{32});
            EXPECT_FALSE(widest.assumed);
        }

        TEST(Small_DeviceProbeFallbacks, ADeviceReportingNoCacheGetsTheAssumedSizeAndIsMarkedAssumed)
        {
            const Device::Detail::Resolved none = Device::Detail::resolve_cache_bytes(0);
            const Device::Detail::Resolved reported = Device::Detail::resolve_cache_bytes(4096);

            EXPECT_EQ(none.value, Device::K_ASSUMED_CACHE_BYTES);
            EXPECT_TRUE(none.assumed);
            EXPECT_EQ(reported.value, std::size_t{4096});
            EXPECT_FALSE(reported.assumed);
        }

        TEST(Medium_DeviceProbeLog, AProbeStatesTheProfileOfTheDeviceOnce)
        {
            const sycl::device device = TestEnvironment::queue(0).get_device();
            const std::string name = device.get_info<sycl::info::device::name>();
            LogCapture capture;

            static_cast<void>(Device::Detail::probe(device));

            EXPECT_EQ(capture.count(SushiLog::LogLevel::Info, "device '" + name + "'"), std::size_t{1});
        }

        TEST(Medium_DeviceProbeLog, AFallbackWarnsOncePerDeviceAndNamesIt)
        {
            const sycl::device device = TestEnvironment::queue(0).get_device();
            const std::string name = device.get_info<sycl::info::device::name>();
            const bool no_sub_group = Device::Detail::resolve_sub_group_size(
                device.get_info<sycl::info::device::sub_group_sizes>()).assumed;
            const bool no_cache = Device::Detail::resolve_cache_bytes(static_cast<std::size_t>(
                device.get_info<sycl::info::device::global_mem_cache_size>())).assumed;
            LogCapture capture;

            static_cast<void>(Device::Detail::probe(device));

            EXPECT_EQ(capture.count(SushiLog::LogLevel::Warn, "lists no sub-group size"),
                      std::size_t{no_sub_group ? 1U : 0U});
            EXPECT_EQ(capture.count(SushiLog::LogLevel::Warn, "reports no cache size"),
                      std::size_t{no_cache ? 1U : 0U});
            EXPECT_EQ(capture.count(SushiLog::LogLevel::Warn, "'" + name + "'"),
                      std::size_t{(no_sub_group ? 1U : 0U) + (no_cache ? 1U : 0U)});
        }
    } // namespace Test
} // namespace SushiBLAS
