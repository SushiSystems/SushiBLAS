/**************************************************************************/
/* test_reduce_1d.cpp                                                     */
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
 * @file test_reduce_1d.cpp
 * @brief Tests the accumulators, the reduction plan and a reduction on the host device.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The element count of the reductions run on the host device.
        constexpr std::size_t K_COUNT = 50000;

        /// @brief The number of partial sums the host-device reductions may use.
        constexpr std::size_t K_PARTIALS = 64;

        /// @brief Returns a CPU profile with the given unit count and group limit.
        static Device::Profile cpu_with(std::size_t units, std::size_t max_group)
        {
            Device::Profile profile;
            profile.is_cpu = true;
            profile.compute_units = units;
            profile.max_work_group_size = max_group;
            return profile;
        }

        TEST(Unit_Reduce1D, FloorPow2RoundsDownToAPowerOfTwo)
        {
            static_assert(Reduce::floor_pow2(1) == 1, "1 is 2^0.");
            static_assert(Reduce::floor_pow2(100) == 64, "100 rounds to 64.");

            EXPECT_EQ(Reduce::floor_pow2(256), std::size_t{256});
            EXPECT_EQ(Reduce::floor_pow2(0), std::size_t{1});
        }

        TEST(Unit_Reduce1D, ACPUFoldsSmallInputsInOneTask)
        {
            const Device::Profile cpu = cpu_with(8, 8192);

            EXPECT_EQ(Reduce::planned_partials(cpu, 0), std::size_t{0});
            EXPECT_EQ(Reduce::planned_partials(cpu, Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS - 1),
                      std::size_t{0});
            EXPECT_TRUE(Reduce::plan(cpu, 100, K_PARTIALS).sequential());
        }

        TEST(Unit_Reduce1D, APlanStaysWithinTheScratchAndTheGroupCap)
        {
            const Device::Profile cpu = cpu_with(64, 64);
            const std::size_t n = std::size_t{1} << 22;

            EXPECT_LE(Reduce::plan(cpu, n, 1000).groups, Reduce::K_MAX_GROUPS);
            EXPECT_EQ(Reduce::plan(cpu, n, 10).groups, std::size_t{10});
            EXPECT_TRUE(Reduce::plan(cpu, n, 1).sequential());
            EXPECT_EQ(Reduce::plan(cpu, n, 10).local, Reduce::group_width(cpu));
        }

        TEST(Unit_Reduce1D, CompensatedSumKeepsTheSmallTermsOfALargeSum)
        {
            Reduce::CompensatedSum<float> sum;
            sum.accumulate(1.0e8f);
            for (int i = 0; i < 1000; ++i)
                sum.accumulate(1.0f);
            sum.accumulate(-1.0e8f);

            EXPECT_EQ(sum.value(), 1000.0f);
        }

        TEST(Unit_Reduce1D, CompensatedSumOfAnInfinityIsThatInfinity)
        {
            Reduce::CompensatedSum<double> sum;
            sum.accumulate(1.0);
            sum.accumulate(std::numeric_limits<double>::infinity());

            EXPECT_EQ(sum.value(), std::numeric_limits<double>::infinity());
        }

        TEST(Unit_Reduce1D, ScaledSumOfSquaresNeitherOverflowsNorUnderflows)
        {
            Reduce::ScaledSumOfSquares<float> big;
            big.accumulate(3.0e30f);
            big.accumulate(4.0e30f);
            Reduce::ScaledSumOfSquares<float> small;
            small.accumulate(3.0e-30f);
            small.accumulate(4.0e-30f);

            EXPECT_NEAR(big.value(), 5.0e30f, 5.0e30f * 1e-6f);
            EXPECT_NEAR(small.value(), 5.0e-30f, 5.0e-30f * 1e-6f);
        }

        TEST(Unit_Reduce1D, ProductExactSumAndComplexSumFoldTheirTerms)
        {
            Reduce::RunningProduct<double> product;
            Reduce::ExactSum<int64_t> exact;
            Reduce::CompensatedComplexSum<std::complex<double>> complex_sum;
            for (int i = 1; i <= 5; ++i)
            {
                product.accumulate(static_cast<double>(i));
                exact.accumulate(i);
                complex_sum.accumulate(std::complex<double>(i, -i));
            }

            EXPECT_EQ(product.value(), 120.0);
            EXPECT_EQ(exact.value(), int64_t{15});
            EXPECT_EQ(complex_sum.value(), std::complex<double>(15.0, -15.0));
        }

        TEST(Unit_Reduce1D, MaxAndMinPropagateNaNAndPreferPositiveZero)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();

            EXPECT_TRUE(std::isnan(Reduce::NaNPropagatingMax<double>::combine(1.0, nan)));
            EXPECT_TRUE(std::isnan(Reduce::NaNPropagatingMin<double>::combine(nan, 1.0)));
            EXPECT_FALSE(std::signbit(Reduce::NaNPropagatingMax<double>::combine(-0.0, 0.0)));
            EXPECT_TRUE(std::signbit(Reduce::NaNPropagatingMin<double>::combine(0.0, -0.0)));
            EXPECT_EQ(Reduce::NaNPropagatingMax<double>::identity(),
                      -std::numeric_limits<double>::infinity());
            EXPECT_EQ(Reduce::NaNPropagatingMin<int32_t>::identity(),
                      std::numeric_limits<int32_t>::max());
        }

        TEST(Unit_Reduce1D, ArgMaxKeepsTheFirstIndexOfATie)
        {
            using Pair = Reduce::IndexedValue<float>;
            Reduce::RunningArgMax<Pair> best;
            best.accumulate(Pair{2.0f, 0});
            best.accumulate(Pair{7.0f, 1});
            best.accumulate(Pair{7.0f, 2});

            EXPECT_EQ(best.value().index, int64_t{1});
            EXPECT_EQ(best.value().value, 7.0f);
        }

        TEST(Unit_Reduce1D, ArgMinTakesTheSmallestValue)
        {
            using Pair = Reduce::IndexedValue<double>;
            Reduce::RunningArgMin<Pair> best;
            best.accumulate(Pair{4.0, 0});
            best.accumulate(Pair{-1.0, 1});
            best.accumulate(Pair{3.0, 2});

            EXPECT_EQ(best.value().index, int64_t{1});
        }

        /** @brief Runs the reductions on the host CPU queue. */
        class Device_Reduce1D : public HostDeviceTest
        {

        };

        TEST_F(Device_Reduce1D, TheStagedAndTheSequentialSumGiveTheExactSum)
        {
            std::vector<float> values(K_COUNT);
            std::size_t exact_sum = 0;
            for (std::size_t i = 0; i < K_COUNT; ++i)
            {
                values[i] = static_cast<float>(i % 7);
                exact_sum += i % 7;
            }
            const SharedArray<float> in(queue(), values);
            SharedArray<float> staged(queue(), 1);
            SharedArray<float> sequential(queue(), 1);
            SharedArray<float> partials(queue(), K_PARTIALS);
            const auto identity = [](float x) { return x; };
            const auto keep = [](float value, std::size_t) { return value; };

            Reduce::reduce<Reduce::CompensatedSum>(queue(), Reduce::Plan{16, 32}, in.get(),
                                                   staged.get(), K_COUNT, partials.get(),
                                                   identity, keep, {}).wait();
            Reduce::reduce<Reduce::CompensatedSum>(queue(), Reduce::Plan{}, in.get(),
                                                   sequential.get(), K_COUNT, partials.get(),
                                                   identity, keep, {}).wait();

            // Each term and partial sum is an integer below 2^24, so both folds are exact.
            const float exact = static_cast<float>(exact_sum);
            EXPECT_EQ(staged.get()[0], exact);
            EXPECT_EQ(sequential.get()[0], exact);
        }

        TEST_F(Device_Reduce1D, FinalizeReceivesTheElementCount)
        {
            const SharedArray<double> in(queue(), std::vector<double>{2.0, 4.0, 6.0, 8.0});
            SharedArray<double> mean(queue(), 1);
            SharedArray<double> partials(queue(), K_PARTIALS);

            Reduce::reduce<Reduce::CompensatedSum>(
                queue(), in.get(), mean.get(), 4, partials.get(), K_PARTIALS,
                [](double x) { return x; },
                [](double value, std::size_t n) { return value / static_cast<double>(n); },
                {}).wait();

            EXPECT_EQ(mean.get()[0], 5.0);
        }
    } // namespace Test
} // namespace SushiBLAS
