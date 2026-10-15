/**************************************************************************/
/* test_centred_reduction.cpp                                             */
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
 * @file test_centred_reduction.cpp
 * @brief Tests the centred reduction dispatcher as a variance, on the reference path and on SYCL.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reductions/centred_reduction.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The dispatcher the variance reduction composes.
        using VarianceDispatch =
            Dispatch::CentredReductionDispatch<Reduce::CompensatedSum, Dispatch::MeanOfSum,
                                               Dispatch::SquaredDeviation,
                                               Dispatch::MeanOfDeviations>;

        /// @brief The element count; a multiple of eight, so the mean of small integers is exact.
        constexpr std::size_t K_COUNT = 4096;

        /** @brief Runs the dispatcher on the host CPU queue. */
        class Device_CentredReduction : public HostDeviceTest
        {

        };

        TEST_F(Device_CentredReduction, TheReferenceVarianceOfOneToFiveIsTwo)
        {
            const std::vector<double> values{1, 2, 3, 4, 5};
            double variance = 0;

            VarianceDispatch{}.reference(values.data(), &variance, 5);

            EXPECT_EQ(variance, 2.0);
        }

        TEST_F(Device_CentredReduction, TheTermsCentreOnTheValueTheyAreGiven)
        {
            const double centre = 3.0;

            EXPECT_EQ(Dispatch::MeanOfSum{}(12.0, std::size_t{4}), 3.0);
            EXPECT_EQ(Dispatch::SquaredDeviation<double>{&centre}(5.0), 4.0);
            EXPECT_EQ(Dispatch::RootOfMeanDeviations<double>{&centre}(36.0, std::size_t{4}), 3.0);
            EXPECT_EQ(Dispatch::ExpBelowCentre<double>{&centre}(3.0), 1.0);
        }

        TEST_F(Device_CentredReduction, TheSYCLVarianceEqualsTheReference)
        {
            const SharedArray<double> in(queue(), small_integers<double>(K_COUNT, 9));
            SharedArray<double> out(queue(), 1);
            const std::size_t capacity =
                Reduce::planned_partials(Device::profile_for(queue()), K_COUNT) + 1;
            SharedArray<double> scratch(queue(), capacity);
            double expected = 0;

            VarianceDispatch{}.reference(in.get(), &expected, static_cast<int64_t>(K_COUNT));
            VarianceDispatch{}(queue(), in.get(), out.get(), static_cast<int64_t>(K_COUNT),
                               Dispatch::ReductionScratch<double>{scratch.get(), scratch.size()},
                               {}).wait();

            EXPECT_EQ(out.get()[0], expected);
        }
    } // namespace Test
} // namespace SushiBLAS
