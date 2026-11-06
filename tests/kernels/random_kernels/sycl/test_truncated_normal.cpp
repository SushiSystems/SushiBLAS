/**************************************************************************/
/* test_truncated_normal.cpp                                              */
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
 * @file test_truncated_normal.cpp
 * @brief Tests the normal distribution function and the truncated draws it inverts.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/sycl/truncated_normal.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Truncated = Kernels::SYCL::Random;

        /// @brief The seed of the draws below.
        constexpr uint64_t K_SEED = 777u;

        /// @brief The rounded operations the bisection and the device erfc may differ by on a value of order one.
        constexpr std::size_t K_BISECTION_OPERATIONS = 1024;

        /// @brief One point of the standard normal distribution function and its value.
        struct CdfRow
        {
                const char* name;
                double z;
                double expected; ///< Phi(z), correctly rounded.
        };

        /// @brief Returns the centre and one point on each side of it.
        static std::vector<CdfRow> cdf_rows()
        {
            return {{"AtZero", 0.0, 0.5},
                    {"AtOne", 1.0, 0.8413447460685429},
                    {"AtMinusTwo", -2.0, 0.02275013194817921}};
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string cdf_row_name(const ::testing::TestParamInfo<CdfRow>& info)
        {
            return info.param.name;
        }

        /// @brief Evaluates the distribution function at one point.
        class Small_TruncatedNormalDistribution : public ::testing::TestWithParam<CdfRow>
        {
        };

        TEST_P(Small_TruncatedNormalDistribution, TheDistributionFunctionHasItsKnownValues)
        {
            const CdfRow& row = GetParam();

            EXPECT_NEAR(Truncated::standard_normal_cdf(row.z), row.expected, absolute_tolerance<double>(1));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TruncatedNormalDistribution, ::testing::ValuesIn(cdf_rows()),
                                 cdf_row_name);

        /// @brief One window, the draws taken from it and the scalar type they are taken in.
        struct WindowRow
        {
                const char* name;
                uint64_t offset;
                uint64_t blocks;
                double lo;
                double hi;
                bool (*all_inside)(const WindowRow&); ///< Draws in the row's scalar type and checks the bounds.
        };

        /// @brief Returns whether every draw of the row's blocks, taken in T, lies in the row's bounds.
        template<typename T>
        static bool every_draw_is_inside(const WindowRow& row)
        {
            for (uint64_t block = 0; block < row.blocks; ++block)
            {
                const auto values = Truncated::draw_truncated_standard_normal<T>(
                    K_SEED, row.offset, block, static_cast<T>(row.lo), static_cast<T>(row.hi));
                const bool inside = std::all_of(values.begin(), values.end(), [&](T value)
                                                { return value >= static_cast<T>(row.lo) &&
                                                         value <= static_cast<T>(row.hi); });
                if (!inside)
                    return false;
            }
            return true;
        }

        /// @brief Returns a wide and a narrow window, in double and in float.
        static std::vector<WindowRow> window_rows()
        {
            return {{"Double_AsymmetricWindow", 0, 200, -1.0, 0.5, &every_draw_is_inside<double>},
                    {"Double_NarrowWindow", 1, 1, -0.01, 0.01, &every_draw_is_inside<double>},
                    {"Float_AsymmetricWindow", 0, 200, -1.0, 0.5, &every_draw_is_inside<float>},
                    {"Float_NarrowWindow", 1, 1, -0.01, 0.01, &every_draw_is_inside<float>}};
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string window_row_name(const ::testing::TestParamInfo<WindowRow>& info)
        {
            return info.param.name;
        }

        /// @brief Draws from one window.
        class Small_TruncatedNormalWindows : public ::testing::TestWithParam<WindowRow>
        {
        };

        TEST_P(Small_TruncatedNormalWindows, EveryDrawLiesInsideItsBounds)
        {
            const WindowRow& row = GetParam();

            EXPECT_TRUE(row.all_inside(row));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TruncatedNormalWindows, ::testing::ValuesIn(window_rows()),
                                 window_row_name);

        namespace
        {
            /// @brief Returns the first values of the stream at offset 3 in [-2, 1], two per block, as the host draws them.
            std::vector<double> host_truncated_draws(std::size_t blocks)
            {
                std::vector<double> host;
                for (std::size_t block = 0; block < blocks; ++block)
                {
                    const auto values =
                        Truncated::draw_truncated_standard_normal<double>(K_SEED, 3, block, -2.0, 1.0);
                    host.insert(host.end(), values.begin(), values.end());
                }
                return host;
            }

            /// @brief Returns the same values as a kernel draws them.
            std::vector<double> device_truncated_draws(sycl::queue& queue, std::size_t blocks)
            {
                SharedArray<double> device(queue, 2 * blocks);
                double* out = device.get();

                queue.parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> i)
                {
                    const auto values =
                        Truncated::draw_truncated_standard_normal<double>(K_SEED, 3, i[0], -2.0, 1.0);
                    out[2 * i[0]] = values[0];
                    out[2 * i[0] + 1] = values[1];
                }).wait();

                return device.values();
            }

            /**
             * @brief Expects each device value to lie within the bisection's step of the host value.
             *
             * erfc may round differently on the device; the bisection then ends a step away.
             */
            void expect_within_the_bisection(const std::vector<double>& device, const std::vector<double>& host)
            {
                ASSERT_EQ(device.size(), host.size());
                for (std::size_t i = 0; i < host.size(); ++i)
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        device[i], host[i], absolute_tolerance<double>(K_BISECTION_OPERATIONS), 1.0)))
                        << "element " << i;
            }
        } // namespace

        /** @brief Draws on the host CPU queue. */
        class Medium_TruncatedNormalOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_TruncatedNormalOnDevice, AKernelDrawsTheValuesTheHostDraws)
        {
            constexpr std::size_t blocks = 64;

            expect_within_the_bisection(device_truncated_draws(queue(), blocks),
                                        host_truncated_draws(blocks));
        }
    } // namespace Test
} // namespace SushiBLAS
