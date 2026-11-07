/**************************************************************************/
/* test_device_meter.cpp                                                  */
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
 * @file test_device_meter.cpp
 * @brief Tests that DeviceMeter produces positive, finite rates on the software device the tests run on.
 * @author Mustafa Garip
 * @see tests/applications/benchmarks/device_ceilings/README.md
 */

#include <cmath>
#include <gtest/gtest.h>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <device_meter.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a short protocol, so a test on a CPU device finishes quickly. */
            Ceilings::Options short_protocol()
            {
                Ceilings::Options options;
                options.runs = 3;
                options.warmup = 1;
                options.min_span_ms = 1.0;
                options.spread_limit = 10.0;
                options.stream_mib = 8;
                return options;
            }
        } // namespace

        TEST(Medium_DeviceMeter, TheF32ComputeCeilingTakesTheProtocolRuns)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            EXPECT_EQ(meter.measure_compute(Ceilings::ComputeKind::F32).runs, 3U);
        }

        TEST(Medium_DeviceMeter, TheF32ComputeCeilingIsFinite)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            EXPECT_TRUE(std::isfinite(meter.measure_compute(Ceilings::ComputeKind::F32).rate));
        }

        TEST(Medium_DeviceMeter, TheF32ComputeCeilingIsPositive)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            EXPECT_GT(meter.measure_compute(Ceilings::ComputeKind::F32).rate, 0.0);
        }

        TEST(Medium_DeviceMeter, TheBandwidthIsFinite)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            EXPECT_TRUE(std::isfinite(meter.measure_bandwidth().rate));
        }

        TEST(Medium_DeviceMeter, TheBandwidthIsPositive)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            EXPECT_GT(meter.measure_bandwidth().rate, 0.0);
        }

        TEST(Medium_DeviceMeter, TheLaunchFloorIsNotNegative)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            EXPECT_GE(meter.measure_launch_floor_ns(), 0.0);
        }

        TEST(Medium_DeviceMeter, F32IsAlwaysSupported)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            EXPECT_TRUE(meter.supports(Ceilings::ComputeKind::F32));
        }
    } // namespace Test
} // namespace SushiBLAS
