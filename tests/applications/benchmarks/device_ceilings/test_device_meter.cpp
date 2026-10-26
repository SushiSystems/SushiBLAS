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

        TEST(Medium_DeviceMeter, TheF32ComputeCeilingIsPositiveAndFinite)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            const Ceilings::Measurement measurement = meter.measure_compute(Ceilings::ComputeKind::F32);

            EXPECT_EQ(measurement.runs, 3U);
            EXPECT_TRUE(std::isfinite(measurement.rate));
            EXPECT_GT(measurement.rate, 0.0);
        }

        TEST(Medium_DeviceMeter, TheBandwidthIsPositiveAndFinite)
        {
            Ceilings::DeviceMeter meter(TestEnvironment::runtime(), short_protocol());

            const Ceilings::Measurement measurement = meter.measure_bandwidth();

            EXPECT_TRUE(std::isfinite(measurement.rate));
            EXPECT_GT(measurement.rate, 0.0);
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

        TEST(Medium_DeviceMeter, EveryComputeKindHasAName)
        {
            EXPECT_STREQ(Ceilings::compute_name(Ceilings::ComputeKind::F16), "f16");
            EXPECT_STREQ(Ceilings::compute_name(Ceilings::ComputeKind::F32), "f32");
            EXPECT_STREQ(Ceilings::compute_name(Ceilings::ComputeKind::F64), "f64");
        }
    } // namespace Test
} // namespace SushiBLAS
