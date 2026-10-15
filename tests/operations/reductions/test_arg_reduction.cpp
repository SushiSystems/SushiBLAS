/**************************************************************************/
/* test_arg_reduction.cpp                                                 */
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
 * @file test_arg_reduction.cpp
 * @brief Tests the arg reduction dispatcher on the reference path and on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reductions/arg_reduction.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The element count; spans several partials on any device.
        constexpr std::size_t K_COUNT = 5003;

        /** @brief Runs the dispatcher on the host CPU queue. */
        class Device_ArgReduction : public HostDeviceTest
        {

        };

        TEST_F(Device_ArgReduction, TheReferenceFindsTheFirstLargestAndSmallest)
        {
            const std::vector<float> values{2.0f, 7.0f, -3.0f, 7.0f, -3.0f};
            int64_t index = -1;

            Dispatch::ArgReductionDispatch<Reduce::RunningArgMax>{}.reference(
                values.data(), &index, static_cast<int64_t>(values.size()));
            EXPECT_EQ(index, int64_t{1});

            Dispatch::ArgReductionDispatch<Reduce::RunningArgMin>{}.reference(
                values.data(), &index, static_cast<int64_t>(values.size()));
            EXPECT_EQ(index, int64_t{2});
        }

        TEST_F(Device_ArgReduction, TheSYCLPathPicksTheIndexTheReferencePicks)
        {
            const SharedArray<double> in(queue(), small_integers<double>(K_COUNT, 5));
            SharedArray<int64_t> out(queue(), 1);
            const std::size_t capacity =
                Reduce::planned_partials(Device::profile_for(queue()), K_COUNT);
            SharedArray<Reduce::IndexedValue<double>> partials(queue(), capacity + 1);
            int64_t expected = -1;

            Dispatch::ArgReductionDispatch<Reduce::RunningArgMax>{}.reference(
                in.get(), &expected, static_cast<int64_t>(K_COUNT));
            Dispatch::ArgReductionDispatch<Reduce::RunningArgMax>{}(
                queue(), in.get(), out.get(), static_cast<int64_t>(K_COUNT),
                Dispatch::ReductionScratch<Reduce::IndexedValue<double>>{partials.get(),
                                                                         partials.size()},
                {}).wait();

            EXPECT_EQ(out.get()[0], expected);
        }
    } // namespace Test
} // namespace SushiBLAS
