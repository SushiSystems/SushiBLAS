/**************************************************************************/
/* test_cpu_queue.cpp                                                     */
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
 * @file test_cpu_queue.cpp
 * @brief Tests that the test environment makes one in-order queue on a CPU device, or none.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/test_runner/cpu_queue.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_CpuQueue, NoDeviceGivesNoQueue)
        {
            EXPECT_FALSE(Detail::make_cpu_queue(std::vector<sycl::device>{}).has_value());
        }

        TEST(Medium_CpuQueue, TheEnvironmentHasAQueueExactlyWhenSYCLReportsACPUDevice)
        {
            const bool has_cpu = !sycl::device::get_devices(sycl::info::device_type::cpu).empty();

            EXPECT_EQ(TestEnvironment::cpu_queue() != nullptr, has_cpu);
        }
    } // namespace Test
} // namespace SushiBLAS
