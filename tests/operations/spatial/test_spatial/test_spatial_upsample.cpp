/**************************************************************************/
/* test_spatial_upsample.cpp                                              */
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
 * @file test_spatial_upsample.cpp
 * @brief Tests SpatialOps::upsample_nearest and upsample_nearest_backward.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
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
            /** @brief Returns count values 0, 1, 2 and so on. */
            std::vector<float> upsample_ramp_values(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
                return values;
            }

            /** @brief Returns a rank-4 NHWC float tensor holding values in memory order. */
            Tensor upsample_image(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c,
                                  const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor({n, h, w, c}, Core::DataType::Float32);
                fill_tensor(t, values);
                return t;
            }

            /** @brief Returns a rank-4 NHWC float tensor holding 0, 1, 2, ... in memory order. */
            Tensor upsample_ramp(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c)
            {
                return upsample_image(engine, n, h, w, c, upsample_ramp_values(n * h * w * c));
            }

            /**
             * @brief Returns the sum of the scale_h x scale_w block of dy that read each input pixel.
             * @param dy The gradient, N x (H * scale_h) x (W * scale_w) x C in memory order.
             * @see tests/operations/spatial/README.md
             */
            std::vector<float> upsample_block_sums(const std::vector<float>& dy, int64_t N,
                                                   int64_t H, int64_t W, int64_t C,
                                                   int64_t scale_h, int64_t scale_w)
            {
                const int64_t out_h = H * scale_h, out_w = W * scale_w;
                std::vector<float> sums(static_cast<std::size_t>(N * H * W * C), 0.0f);
                for (int64_t n = 0; n < N; ++n)
                    for (int64_t h = 0; h < H; ++h)
                        for (int64_t w = 0; w < W; ++w)
                            for (int64_t c = 0; c < C; ++c)
                                for (int64_t dh = 0; dh < scale_h; ++dh)
                                    for (int64_t dw = 0; dw < scale_w; ++dw)
                                    {
                                        const int64_t row = h * scale_h + dh;
                                        const int64_t col = w * scale_w + dw;
                                        sums[static_cast<std::size_t>(((n * H + h) * W + w) * C + c)] +=
                                            dy[static_cast<std::size_t>(
                                                ((n * out_h + row) * out_w + col) * C + c)];
                                    }
                return sums;
            }

            /** @brief Returns count values of a reproducible pattern in -11 .. 11. */
            std::vector<float> upsample_pattern(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>((i * 37) % 23) - 11.0f;
                return values;
            }

            /** @brief Names a scale and the message its refusal carries. */
            struct ScaleRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The row scale. */
                int64_t scale_h;

                /** @brief The column scale. */
                int64_t scale_w;

                /** @brief The whole refusal message. */
                std::string message;
            };

            /** @brief Names a row by its name. */
            std::string upsample_row_name(const ::testing::TestParamInfo<ScaleRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Upsamples on the host engine. */
        class Small_SpatialUpsampleTest : public HostEngineTest
        {
        };

        /** @brief Records an upsample with one malformed scale on a host engine. */
        class Small_SpatialUpsampleScaleTest : public HostEngineTest,
                                               public ::testing::WithParamInterface<ScaleRow>
        {
        };

        /** @brief Upsamples on the device and on the reference path. */
        class Medium_SpatialUpsampleTest : public SushiBLASTest
        {
        };

        TEST_F(Small_SpatialUpsampleTest, EachInputPixelBecomesAScaleByScaleBlock)
        {
            Tensor in = upsample_image(*engine, 1, 2, 2, 1, {0, 1,
                                                             2, 3});
            Tensor out = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).upsample_nearest(in, out, 2, 2);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, {0, 0, 1, 1,
                                      0, 0, 1, 1,
                                      2, 2, 3, 3,
                                      2, 2, 3, 3});
        }

        TEST_F(Small_SpatialUpsampleTest, AnisotropicScalesAreIndependent)
        {
            // Each pixel becomes a 2-row by 3-column block, so the 2x2 input becomes 4x6.
            Tensor in = upsample_image(*engine, 1, 2, 2, 1, {1, 2,
                                                             3, 4});
            Tensor out = engine->create_tensor({1, 4, 6, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).upsample_nearest(in, out, 2, 3);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, {1, 1, 1, 2, 2, 2,
                                      1, 1, 1, 2, 2, 2,
                                      3, 3, 3, 4, 4, 4,
                                      3, 3, 3, 4, 4, 4});
        }

        TEST_F(Small_SpatialUpsampleTest, AUnitScaleIsACopy)
        {
            Tensor in = upsample_ramp(*engine, 2, 3, 4, 2);
            Tensor out = engine->create_tensor({2, 3, 4, 2}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).upsample_nearest(in, out, 1, 1);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, upsample_ramp_values(in.num_elements));
        }

        TEST_F(Small_SpatialUpsampleTest, TheBackwardSumsTheBlockThatReadEachInput)
        {
            // The block is summed index by index. See tests/operations/spatial/README.md.
            const int64_t N = 2, H = 3, W = 2, C = 2, scale_h = 2, scale_w = 3;
            Tensor dy = upsample_ramp(*engine, N, H * scale_h, W * scale_w, C);
            Tensor dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).upsample_nearest_backward(dy, dx, scale_h, scale_w);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, upsample_block_sums(upsample_ramp_values(dy.num_elements), N, H,
                                                        W, C, scale_h, scale_w));
        }

        TEST_P(Small_SpatialUpsampleScaleTest, AScaleThatIsNotPositiveIsRefusedNamingTheOperand)
        {
            const ScaleRow& row = GetParam();
            Tensor in = upsample_image(*engine, 1, 2, 2, 1, {1, 2,
                                                             3, 4});
            Tensor out = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            try
            {
                SushiBLAS::SpatialOps(recorder()).upsample_nearest(in, out, row.scale_h,
                                                                   row.scale_w);
                ADD_FAILURE() << "a scale of " << row.scale_h << " by " << row.scale_w
                              << " was accepted";
            }
            catch (const std::invalid_argument& error)
            {
                EXPECT_EQ(std::string(error.what()), row.message);
            }
        }

        INSTANTIATE_TEST_SUITE_P(
            Scales, Small_SpatialUpsampleScaleTest,
            ::testing::Values(
                ScaleRow{"ZeroRowScale", 0, 2,
                         "spatial.upsample_nearest: operand 'scale_h' is 0; a scale must be positive"},
                ScaleRow{"NegativeColumnScale", 2, -3,
                         "spatial.upsample_nearest: operand 'scale_w' is -3; a scale must be positive"}),
            upsample_row_name);

        TEST_F(Medium_SpatialUpsampleTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            const int64_t N = 2, H = 5, W = 3, C = 3, scale_h = 3, scale_w = 2;
            Tensor in = upsample_image(*engine, N, H, W, C, upsample_pattern(N * H * W * C));
            Tensor from_reference =
                engine->create_tensor({N, H * scale_h, W * scale_w, C}, Core::DataType::Float32);
            Tensor from_device =
                engine->create_tensor({N, H * scale_h, W * scale_w, C}, Core::DataType::Float32);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).upsample_nearest(in, from_reference, scale_h, scale_w);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).upsample_nearest(in, from_device, scale_h, scale_w);
            recording->run().value();

            expect_tensor_equal(from_device, read_tensor<float>(from_reference));
        }
    } // namespace Test
} // namespace SushiBLAS
