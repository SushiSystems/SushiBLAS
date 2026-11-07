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
 * @file test_device_policy.cpp
 * @brief Tests the cache blocking, the group sizes, the 1D grid and the vector width of a profile.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiLog/log/log_capture.hpp>
#include <SushiBLAS/test_runner/captured_record_count.hpp>
#include <SushiBLAS/test_runner/scoped_log_level.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Names the group limit of a GPU profile and the work-group size it plans.
            struct GpuGroupRow
            {
                /// @brief Holds the largest work-group the device accepts.
                std::size_t max_group;
                /// @brief Holds the work-group size the policy plans.
                std::size_t expected;
            };

            /// @brief Names an element count and the flat work-group size a CPU plans for it.
            struct CpuGroupRow
            {
                /// @brief Holds the element count.
                std::size_t count;
                /// @brief Holds the flat work-group size the policy plans.
                std::size_t expected;
            };

            /// @brief Names an element count handed to a GPU's flat work-group size.
            struct GpuFlatRow
            {
                /// @brief Holds the element count.
                std::size_t count;
            };

            /// @brief Names a local tile request, and whether a profile with 48 KiB and 256 hosts it.
            struct LocalTileRow
            {
                /// @brief Holds the local memory the tile asks for, in bytes.
                std::size_t bytes;
                /// @brief Holds the work-group size the tile asks for.
                std::size_t group;
                /// @brief Holds whether the profile hosts the tile.
                bool fits;
            };

            /// @brief Names an architecture, the local memory the device reports and the table's entry.
            struct LocalMemoryRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the architecture the table is asked about.
                Device::Architecture architecture;
                /// @brief Holds the local memory the device reports, in bytes.
                std::size_t reported;
                /// @brief Holds the vendor's documented local memory per compute unit, in bytes (NVIDIA CUDA Programming Guide compute-capability table: 100 KB sm_86, 164 KB sm_80, 228 KB sm_90; Intel PVC: 128 KB per Xe-core).
                std::size_t expected;
            };

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

            /// @brief Returns whether an architecture is one of the NVIDIA generations.
            bool is_nvidia_architecture(Device::Architecture architecture)
            {
                return architecture >= Device::Architecture::NvidiaSm70 &&
                       architecture <= Device::Architecture::NvidiaSm90;
            }

            /// @brief Returns whether a device is a CUDA device.
            bool is_cuda(const sycl::queue& queue)
            {
                return queue.get_device().get_backend() == sycl::backend::ext_oneapi_cuda;
            }

            /// @brief Returns whether a device is a CUDA device of architecture sm_86.
            bool is_sm86_cuda(const sycl::queue& queue)
            {
#if defined(SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE)
                namespace Experimental = sycl::ext::oneapi::experimental;
                return is_cuda(queue) &&
                       queue.get_device().get_info<Experimental::info::device::architecture>() ==
                           Experimental::architecture::nvidia_gpu_sm_86;
#else
                static_cast<void>(queue);
                return false;
#endif
            }

            /// @brief Returns the indices of the devices of the shared runtime that satisfy a predicate.
            template <typename Predicate>
            std::vector<std::size_t> devices_where(Predicate predicate)
            {
                std::vector<std::size_t> indices;
                for (std::size_t index = 0; index < TestEnvironment::device_count(); ++index)
                {
                    if (predicate(TestEnvironment::queue(index)))
                        indices.push_back(index);
                }
                return indices;
            }

            /// @brief Runs a check on the profile of each listed device.
            template <typename Check>
            void on_each_profile(const std::vector<std::size_t>& indices, Check check)
            {
                for (const std::size_t index : indices)
                    check(Device::profile_for(TestEnvironment::queue(index)));
            }

            /// @brief Returns the indices of every device of the shared runtime.
            std::vector<std::size_t> every_device()
            {
                return devices_where([](const sycl::queue&) { return true; });
            }

            /// @brief Returns whether a device lists no usable sub-group width.
            bool lists_no_sub_group(const std::vector<std::size_t>& widths)
            {
                return widths.empty() || *std::max_element(widths.begin(), widths.end()) == 0;
            }

            /// @brief Names a test case after the row it runs.
            template <typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Tests that the blocking budgets are fractions of the cache size. */
        TEST(Small_DevicePolicy, CacheBlockingScalesWithTheCache)
        {
            const Device::CacheBlocking blocking = Device::cache_blocking_for(1024 * 1024);

            EXPECT_EQ(blocking.outer_bytes, std::size_t{256 * 1024});
            EXPECT_EQ(blocking.stream_bytes, std::size_t{512 * 1024});
            EXPECT_EQ(blocking.min_operand_bytes, std::size_t{1024 * 1024});
            EXPECT_EQ(blocking.inner_bytes, Device::CacheBlocking{}.inner_bytes);
        }

        /** @brief Tests that a default blocking is the blocking of the assumed cache. */
        TEST(Small_DevicePolicy, TheDefaultBlockingIsTheBlockingOfTheAssumedCache)
        {
            const Device::CacheBlocking assumed =
                Device::cache_blocking_for(Device::K_ASSUMED_CACHE_BYTES);
            const Device::CacheBlocking fallback{};

            EXPECT_EQ(assumed.outer_bytes, fallback.outer_bytes);
            EXPECT_EQ(assumed.stream_bytes, fallback.stream_bytes);
            EXPECT_EQ(assumed.min_operand_bytes, fallback.min_operand_bytes);
        }

        /// @brief Runs one group limit against a GPU of ten units and 32-wide sub-groups.
        class Small_DevicePolicyGpuGroup : public ::testing::TestWithParam<GpuGroupRow>
        {
        };

        /** @brief Tests that a GPU group is a whole number of sub-groups under its target. */
        TEST_P(Small_DevicePolicyGpuGroup, AGPUGroupIsAWholeNumberOfSubGroupsUnderItsTarget)
        {
            EXPECT_EQ(Device::work_group_size(gpu_profile(10, GetParam().max_group, 32)),
                      GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_DevicePolicyGpuGroup,
                                 ::testing::Values(GpuGroupRow{1024, 256}, GpuGroupRow{100, 96},
                                                   GpuGroupRow{16, 1}));

        /// @brief Runs one element count against a CPU of four units and a group limit of 8192.
        class Small_DevicePolicyCpuGroup : public ::testing::TestWithParam<CpuGroupRow>
        {
        };

        /** @brief Tests that a CPU group gives each unit its share, above the floor and under the limit. */
        TEST_P(Small_DevicePolicyCpuGroup, ACPUGroupGivesEachUnitItsShareAboveTheFloor)
        {
            EXPECT_EQ(Device::flat_work_group_size(cpu_profile(4, 8192), GetParam().count),
                      GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_DevicePolicyCpuGroup,
            ::testing::Values(CpuGroupRow{100, Device::K_CPU_WORK_GROUP_TARGET},
                              CpuGroupRow{4000, 1000}, CpuGroupRow{1 << 20, 8192}));

        /// @brief Runs one element count against a GPU's flat work-group size.
        class Small_DevicePolicyGpuFlatGroup : public ::testing::TestWithParam<GpuFlatRow>
        {
        };

        /** @brief Tests that a GPU's flat group size does not depend on the element count. */
        TEST_P(Small_DevicePolicyGpuFlatGroup, AGPUFlatGroupIgnoresTheElementCount)
        {
            const Device::Profile gpu = gpu_profile(10, 1024, 32);

            EXPECT_EQ(Device::flat_work_group_size(gpu, GetParam().count),
                      Device::work_group_size(gpu));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_DevicePolicyGpuFlatGroup,
                                 ::testing::Values(GpuFlatRow{7}, GpuFlatRow{1 << 24}));

        /** @brief Tests that a grid covers the elements in whole groups. */
        TEST(Small_DevicePolicy, AGridCoversTheElementsInWholeGroups)
        {
            const Device::Grid1D grid = Device::plan_1d(cpu_profile(4, 8192), 4000);

            EXPECT_EQ(grid.local_size, std::size_t{1000});
            EXPECT_EQ(grid.global_size % grid.local_size, std::size_t{0});
            EXPECT_GE(grid.global_size, std::size_t{4000});
        }

        /** @brief Tests that a GPU grid stops at the groups that are resident at once. */
        TEST(Small_DevicePolicy, AGPUGridStopsAtTheResidentGroups)
        {
            const Device::Profile gpu = gpu_profile(2, 1024, 32);
            const Device::Grid1D grid = Device::plan_1d(gpu, std::size_t{1} << 24);

            EXPECT_EQ(grid.groups(), 2 * Device::K_WAVES_PER_COMPUTE_UNIT);
        }

        /** @brief Tests that unrolling divides the groups a grid needs. */
        TEST(Small_DevicePolicy, UnrollingDividesTheGroupsAGridNeeds)
        {
            const Device::Profile cpu = cpu_profile(1, 64);

            EXPECT_EQ(Device::plan_1d(cpu, 640, 1).groups(), std::size_t{10});
            EXPECT_EQ(Device::plan_1d(cpu, 640, 4).groups(), std::size_t{3});
            EXPECT_EQ(Device::plan_1d(cpu, 0).groups(), std::size_t{1});
        }

        /** @brief Tests that a grid's nd_range carries the grid's sizes. */
        TEST(Small_DevicePolicy, TheGridNdRangeMatchesItsSizes)
        {
            const Device::Grid1D grid{512, 64};
            const sycl::nd_range<1> range = grid.nd_range();

            EXPECT_EQ(range.get_global_range()[0], std::size_t{512});
            EXPECT_EQ(range.get_local_range()[0], std::size_t{64});
        }

        /** @brief Tests that a vector fills one transaction on a GPU and is one element on a CPU. */
        TEST(Small_DevicePolicy, VectorWidthFillsOneTransactionOnAGPUOnly)
        {
            const Device::Profile gpu = gpu_profile(10, 1024, 32);

            EXPECT_EQ(Device::vector_width(gpu, 4), std::size_t{4});
            EXPECT_EQ(Device::vector_width(gpu, 2), std::size_t{8});
            EXPECT_EQ(Device::vector_width(gpu, 16), std::size_t{1});
            EXPECT_EQ(Device::vector_width(gpu, 0), std::size_t{1});
            EXPECT_EQ(Device::vector_width(cpu_profile(4, 64), 4), std::size_t{1});
        }

        /// @brief Runs one local tile request against a profile with 48 KiB of local memory and 256 items.
        class Small_DevicePolicyLocalTile : public ::testing::TestWithParam<LocalTileRow>
        {
        };

        /** @brief Tests that a local tile fits only where the memory and the group both suffice. */
        TEST_P(Small_DevicePolicyLocalTile, ALocalTileFitsWhenMemoryAndGroupBothSuffice)
        {
            Device::Profile profile = gpu_profile(10, 256, 32);
            profile.local_mem_bytes = 48 * 1024;

            EXPECT_EQ(Device::hosts_local_tile(profile, {GetParam().bytes, GetParam().group}),
                      GetParam().fits);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_DevicePolicyLocalTile,
                                 ::testing::Values(LocalTileRow{48 * 1024, 256, true},
                                                   LocalTileRow{1024, 256, true},
                                                   LocalTileRow{48 * 1024 + 1, 256, false},
                                                   LocalTileRow{1024, 257, false}));

        /** @brief Tests that profile_for returns one cached profile per device. */
        TEST(Medium_DevicePolicy, ProfileForKeepsOneProfilePerDevice)
        {
            const sycl::queue& queue = TestEnvironment::queue(0);

            EXPECT_EQ(&Device::profile_for(queue), &Device::profile_for(queue.get_device()));
        }

        /** @brief Tests that the probed fields agree with what the device reports. */
        TEST(Medium_DevicePolicy, TheProbedProfileAgreesWithTheDevice)
        {
            const sycl::queue& queue = TestEnvironment::queue(0);
            const Device::Profile& profile = Device::profile_for(queue);

            EXPECT_EQ(profile.is_cpu, queue.get_device().is_cpu());
            EXPECT_GE(profile.compute_units, std::size_t{1});
            EXPECT_GE(profile.max_work_group_size, std::size_t{1});
            EXPECT_GE(profile.sub_group_size, std::size_t{1});
            EXPECT_EQ(profile.blocking.min_operand_bytes, profile.cache_bytes);
        }

        /** @brief Tests that a device offers the half-precision matrix only on the CUDA backend. */
        TEST(Medium_DevicePolicy, OnlyACUDADeviceOffersTheHalfMatrix)
        {
            const sycl::queue& queue = TestEnvironment::queue(0);

            EXPECT_TRUE(is_cuda(queue) || !Device::profile_for(queue).supports_half_matrix);
        }

        /// @brief Runs a test on a CUDA device, and skips where the first device is not one.
        class Large_DevicePolicyCuda : public ::testing::Test
        {
            protected:
                /** @brief Skips the test when the first device is not a CUDA device. */
                void SetUp() override
                {
                    if (!is_cuda(TestEnvironment::queue(0)))
                        GTEST_SKIP() << "the device is not a CUDA device";
                }
        };

        /** @brief Tests that a CUDA device offers the 16 x 16 x 16 half-precision matrix. */
        TEST_F(Large_DevicePolicyCuda, ACUDADeviceOffersTheHalfMatrixOfSixteen)
        {
            // Every card from sm_70 on lists 16 x 16 x 16 for HALF x HALF -> FLOAT32.
            EXPECT_TRUE(Device::profile_for(TestEnvironment::queue(0)).supports_half_matrix);
        }

        /// @brief Runs one architecture against the local-memory table.
        class Small_DevicePolicyLocalMemory : public ::testing::TestWithParam<LocalMemoryRow>
        {
        };

        /** @brief Tests that a table entry replaces what the device reports. */
        TEST_P(Small_DevicePolicyLocalMemory, ALocalMemoryTableEntryReplacesWhatTheDeviceReports)
        {
            EXPECT_EQ(Device::local_mem_per_compute_unit_of(GetParam().architecture,
                                                            GetParam().reported),
                      GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_DevicePolicyLocalMemory,
            ::testing::Values(
                LocalMemoryRow{"NvidiaSm86", Device::Architecture::NvidiaSm86, 48 * 1024,
                               100 * 1024},
                LocalMemoryRow{"NvidiaSm80", Device::Architecture::NvidiaSm80, 0, 164 * 1024},
                LocalMemoryRow{"NvidiaSm90", Device::Architecture::NvidiaSm90, 0, 228 * 1024},
                LocalMemoryRow{"IntelPVC", Device::Architecture::IntelPVC, 0, 128 * 1024}),
            row_name<LocalMemoryRow>);

        /** @brief Tests that an unknown architecture passes the reported local memory through. */
        TEST(Small_DevicePolicy, AnUnknownArchitectureKeepsTheReportedLocalMemory)
        {
            EXPECT_EQ(Device::local_mem_per_compute_unit_of(Device::Architecture::Unknown, 32768),
                      std::size_t{32768});
        }

        /** @brief Tests that the local memory of each device is what the table gives its architecture. */
        TEST(Medium_DevicePolicy, EveryDeviceTakesItsLocalMemoryFromTheTable)
        {
            on_each_profile(every_device(), [](const Device::Profile& profile)
            {
                // The table function is the named oracle: the probe must route through it.
                EXPECT_EQ(profile.local_mem_per_compute_unit,
                          Device::local_mem_per_compute_unit_of(profile.architecture,
                                                                profile.local_mem_bytes));
            });
        }

        /** @brief Tests that a device of an NVIDIA architecture reports the NVIDIA vendor. */
        TEST(Medium_DevicePolicy, ANvidiaArchitectureHasTheNvidiaVendor)
        {
            on_each_profile(every_device(), [](const Device::Profile& profile)
            {
                EXPECT_TRUE(!is_nvidia_architecture(profile.architecture) ||
                            profile.vendor == Device::Vendor::Nvidia);
            });
        }

        /** @brief Tests that a CPU device reports no NVIDIA architecture. */
        TEST(Medium_DevicePolicy, ACPUDeviceReportsNoNvidiaArchitecture)
        {
            on_each_profile(every_device(), [](const Device::Profile& profile)
            {
                EXPECT_TRUE(!profile.is_cpu || !is_nvidia_architecture(profile.architecture));
            });
        }

        /// @brief Runs a test on the sm_86 CUDA devices, and skips where there is none.
        class Medium_DevicePolicySm86 : public ::testing::Test
        {
            protected:
                /** @brief Skips the test when no sm_86 CUDA device is present. */
                void SetUp() override
                {
                    if (devices_where(is_sm86_cuda).empty())
                        GTEST_SKIP() << "no CUDA device of architecture sm_86 is present";
                }
        };

        /** @brief Tests that an sm_86 CUDA device is read as NvidiaSm86 with its local memory. */
        TEST_F(Medium_DevicePolicySm86, ASm86CUDADeviceIsReadAsNvidiaSm86)
        {
            on_each_profile(devices_where(is_sm86_cuda), [](const Device::Profile& profile)
            {
                EXPECT_EQ(profile.architecture, Device::Architecture::NvidiaSm86);
                EXPECT_EQ(profile.vendor, Device::Vendor::Nvidia);
                EXPECT_EQ(profile.local_mem_per_compute_unit, std::size_t{100 * 1024});
            });
        }

        /** @brief Tests that a listing without a sub-group gets width one and is marked assumed. */
        TEST(Small_DeviceProbeFallbacks, ADeviceListingNoSubGroupGetsWidthOneAndIsMarkedAssumed)
        {
            const Device::Detail::Resolved none = Device::Detail::resolve_sub_group_size({});
            const Device::Detail::Resolved zero = Device::Detail::resolve_sub_group_size({0});

            EXPECT_EQ(none.value, std::size_t{1});
            EXPECT_TRUE(none.assumed);
            EXPECT_EQ(zero.value, std::size_t{1});
            EXPECT_TRUE(zero.assumed);
        }

        /** @brief Tests that a listing of sub-groups keeps the widest and assumes nothing. */
        TEST(Small_DeviceProbeFallbacks, ADeviceListingSubGroupsKeepsTheWidestAndAssumesNothing)
        {
            const Device::Detail::Resolved widest =
                Device::Detail::resolve_sub_group_size({16, 32, 8});

            EXPECT_EQ(widest.value, std::size_t{32});
            EXPECT_FALSE(widest.assumed);
        }

        /** @brief Tests that a device reporting no cache gets the assumed size and is marked assumed. */
        TEST(Small_DeviceProbeFallbacks, ADeviceReportingNoCacheGetsTheAssumedSizeAndIsMarkedAssumed)
        {
            const Device::Detail::Resolved none = Device::Detail::resolve_cache_bytes(0);
            const Device::Detail::Resolved reported = Device::Detail::resolve_cache_bytes(4096);

            EXPECT_EQ(none.value, Device::K_ASSUMED_CACHE_BYTES);
            EXPECT_TRUE(none.assumed);
            EXPECT_EQ(reported.value, std::size_t{4096});
            EXPECT_FALSE(reported.assumed);
        }

        /** @brief Tests that a probe logs one INFO line naming the device. */
        TEST(Medium_DeviceProbeLog, AProbeStatesTheProfileOfTheDeviceOnce)
        {
            const sycl::device device = TestEnvironment::queue(0).get_device();
            const std::string name = device.get_info<sycl::info::device::name>();
            const ScopedLogLevel level(SushiLog::LogLevel::Info);
            const SushiLog::LogCapture capture;

            static_cast<void>(Device::Detail::probe(device));

            EXPECT_EQ(count_records(capture.records(), SushiLog::LogLevel::Info,
                                    "device '" + name + "'"),
                      std::size_t{1});
        }

        /** @brief Tests that a probe warns once for each fallback it takes and names the device. */
        TEST(Medium_DeviceProbeLog, AFallbackWarnsOnceAndNamesTheDevice)
        {
            const sycl::device device = TestEnvironment::queue(0).get_device();
            const std::string name = device.get_info<sycl::info::device::name>();
            // The oracle reads the device itself, so it does not repeat the fallback rules.
            const bool no_sub_group = lists_no_sub_group(
                device.get_info<sycl::info::device::sub_group_sizes>());
            const bool no_cache =
                device.get_info<sycl::info::device::global_mem_cache_size>() == 0;
            const SushiLog::LogCapture capture;

            static_cast<void>(Device::Detail::probe(device));
            const std::vector<SushiLog::CapturedRecord> records = capture.records();

            EXPECT_EQ(count_records(records, SushiLog::LogLevel::Warn, "lists no sub-group size"),
                      std::size_t{no_sub_group ? 1U : 0U});
            EXPECT_EQ(count_records(records, SushiLog::LogLevel::Warn, "reports no cache size"),
                      std::size_t{no_cache ? 1U : 0U});
            EXPECT_EQ(count_records(records, SushiLog::LogLevel::Warn, "'" + name + "'"),
                      std::size_t{(no_sub_group ? 1U : 0U) + (no_cache ? 1U : 0U)});
        }
    } // namespace Test
} // namespace SushiBLAS
