/**************************************************************************/
/* test_spatial_pool_avg.cpp                                              */
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
 * @file test_spatial_pool_avg.cpp
 * @brief Tests SpatialOps::pool_avg and pool_avg_backward.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a rank-4 NHWC float tensor holding values in memory order. */
            Tensor pool_avg_image(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c,
                                  const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor({n, h, w, c}, Core::DataType::Float32);
                fill_tensor(t, values);
                return t;
            }

            /** @brief Returns count values of a reproducible pattern in -11 .. 11. */
            std::vector<float> pool_avg_pattern(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>((i * 37) % 23) - 11.0f;
                return values;
            }

            /** @brief Returns a 2 x 2 window of stride 2. */
            Spatial::ConvGeometry pool_avg_tiles()
            {
                Spatial::ConvGeometry g;
                g.kernel_h = 2;
                g.kernel_w = 2;
                g.stride_h = 2;
                g.stride_w = 2;
                return g;
            }
        } // namespace

        /** @brief Averages windows on the host engine. */
        class Small_SpatialPoolAvgTest : public HostEngineTest
        {
        };

        /** @brief Averages windows on the device and on the reference path. */
        class Medium_SpatialPoolAvgTest : public SushiBLASTest
        {
        };

        /**
         * @brief Tests that the average divides by the whole window area, padded taps included.
         * @see tests/operations/spatial/README.md
         */
        TEST_F(Small_SpatialPoolAvgTest, TheAverageIsOverTheWholeWindowIncludingPaddedTaps)
        {
            Tensor in = pool_avg_image(*engine, 1, 2, 2, 1, {4, 8,
                                                             12, 16});
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;
            Tensor out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).pool_avg(in, out, g);
            ASSERT_TRUE(recording->run_succeeds());

            // The windows sum 4, 4 + 8, 4 + 12 and 4 + 8 + 12 + 16, each over 4. See the test README.
            expect_tensor_equal(out, {1, 3, 4, 10});
        }

        TEST_F(Small_SpatialPoolAvgTest, AnUnpaddedPoolIsThePlainMean)
        {
            Tensor in = pool_avg_image(*engine, 1, 4, 4, 1, {1, 3, 2, 0,
                                                             4, 2, 1, 5,
                                                             0, 7, 6, 2,
                                                             8, 1, 3, 9});
            Tensor out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).pool_avg(in, out, pool_avg_tiles());
            ASSERT_TRUE(recording->run_succeeds());

            // (1+3+4+2)/4, (2+0+1+5)/4, (0+7+8+1)/4 and (6+2+3+9)/4.
            expect_tensor_equal(out, {2.5f, 2.0f, 4.0f, 5.0f});
        }

        TEST_F(Small_SpatialPoolAvgTest, TheBackwardSpreadsEachGradientOverItsWholeWindow)
        {
            Tensor dy = pool_avg_image(*engine, 1, 2, 2, 1, {4, 8,
                                                             12, 16});
            Tensor dx = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).pool_avg_backward(dy, dx, pool_avg_tiles());
            ASSERT_TRUE(recording->run_succeeds());

            // Non-overlapping quadrants, each receiving its window's dy/4.
            expect_tensor_equal(dx, {1, 1, 2, 2,
                                     1, 1, 2, 2,
                                     3, 3, 4, 4,
                                     3, 3, 4, 4});
        }

        TEST_F(Small_SpatialPoolAvgTest, OverlappingWindowsSumTheirShares)
        {
            // The centre pixel is covered by all four windows. See the test README.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            Tensor dy = pool_avg_image(*engine, 1, 2, 2, 1, {1, 2,
                                                             3, 4});
            Tensor dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).pool_avg_backward(dy, dx, g);
            ASSERT_TRUE(recording->run_succeeds());

            // The shares are 0.25, 0.5, 0.75 and 1; a pixel adds the shares of the windows over it.
            expect_tensor_equal(dx, {0.25f, 0.75f, 0.5f,
                                     1.0f, 2.5f, 1.5f,
                                     0.75f, 1.75f, 1.0f});
        }

        TEST_F(Small_SpatialPoolAvgTest, TheBackwardOfAConstantForwardIsUniform)
        {
            // Every pixel is covered by exactly R*S windows. See the test README.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;
            Tensor dy = pool_avg_image(*engine, 1, 4, 4, 1, std::vector<float>(16, 1.0f));
            Tensor dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).pool_avg_backward(dy, dx, g);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, std::vector<float>(9, 1.0f));
        }

        TEST_F(Medium_SpatialPoolAvgTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            const int64_t N = 2, H = 7, W = 5, C = 3;
            Tensor in = pool_avg_image(*engine, N, H, W, C, pool_avg_pattern(N * H * W * C));

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 1;
            g.dilation_h = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(H, 1, 1, 2, 3, 2);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 2, 1);
            Tensor from_reference =
                engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            Tensor from_device =
                engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).pool_avg(in, from_reference, g);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).pool_avg(in, from_device, g);
            recording->run().value();

            expect_tensor_equal(from_device, read_tensor<float>(from_reference));
        }
    } // namespace Test
} // namespace SushiBLAS
