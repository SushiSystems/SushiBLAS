/**************************************************************************/
/* test_spatial_col2im.cpp                                                */
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
 * @file test_spatial_col2im.cpp
 * @brief Tests SpatialOps::col2im against overlap counts and the reference kernel.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
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
            /**
             * @brief Returns how many windows read each pixel of an N x H x W x C image.
             *
             * The count walks every tap of every window and adds one where the tap lands in the
             * image, so it shares no index rule with the kernel. See the test README.
             */
            std::vector<float> overlap_counts(int64_t N, int64_t H, int64_t W, int64_t C,
                                              const Spatial::ConvGeometry& g, int64_t out_h,
                                              int64_t out_w)
            {
                std::vector<float> counts(static_cast<std::size_t>(N * H * W * C), 0.0f);
                for (int64_t n = 0; n < N; ++n)
                    for (int64_t ho = 0; ho < out_h; ++ho)
                        for (int64_t wo = 0; wo < out_w; ++wo)
                            for (int64_t r = 0; r < g.kernel_h; ++r)
                                for (int64_t s = 0; s < g.kernel_w; ++s)
                                {
                                    const int64_t h = ho * g.stride_h - g.pad_top + r * g.dilation_h;
                                    const int64_t w = wo * g.stride_w - g.pad_left + s * g.dilation_w;
                                    if (h < 0 || h >= H || w < 0 || w >= W)
                                        continue;
                                    for (int64_t ch = 0; ch < C; ++ch)
                                        counts[static_cast<std::size_t>(((n * H + h) * W + w) * C + ch)] += 1.0f;
                                }
                return counts;
            }

            /** @brief Returns count values that cycle through 0 .. modulus - 1, less offset. */
            std::vector<float> cycling_values(int64_t count, int64_t modulus, float offset)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>(i % modulus) - offset;
                return values;
            }

            /** @brief Returns a 2 x 2 window of stride 2. */
            Spatial::ConvGeometry two_by_two_stride_two()
            {
                Spatial::ConvGeometry g;
                g.kernel_h = 2;
                g.kernel_w = 2;
                g.stride_h = 2;
                g.stride_w = 2;
                return g;
            }
        } // namespace

        /** @brief Folds column matrices on the host engine. */
        class Small_SpatialCol2ImTest : public HostEngineTest
        {
            protected:
                /** @brief A rank-2 column matrix whose every entry is @p value. */
                Tensor filled_cols(int64_t rows, int64_t width, float value)
                {
                    Tensor t = engine->create_tensor({rows, width}, Core::DataType::Float32);
                    fill_tensor(t, std::vector<float>(static_cast<std::size_t>(t.num_elements), value));
                    return t;
                }
        };

        /** @brief Folds column matrices on the device and on the reference path. */
        class Medium_SpatialCol2ImTest : public SushiBLASTest
        {
        };

        TEST_F(Small_SpatialCol2ImTest, EachPixelReceivesOncePerWindowThatReadIt)
        {
            // All-ones columns give a pixel's overlap count. See the test README.
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t N = 2, H = 4, W = 5, C = 2;
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);

            Tensor cols = filled_cols(N * out_h * out_w, 3 * 3 * C, 1.0f);
            Tensor dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).col2im(cols, dx, g, out_h, out_w);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, overlap_counts(N, H, W, C, g, out_h, out_w));
        }

        TEST_F(Small_SpatialCol2ImTest, StrideSelectsTheRightContributors)
        {
            // With stride 2 each pixel has exactly one window. See the test README.
            const Spatial::ConvGeometry g = two_by_two_stride_two();
            const int64_t H = 4, W = 4, C = 1;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 2);
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 2);

            Tensor cols = filled_cols(out_h * out_w, 2 * 2 * C, 1.0f);
            Tensor dx = engine->create_tensor({1, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).col2im(cols, dx, g, out_h, out_w);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, std::vector<float>(16, 1.0f));
        }

        TEST_F(Small_SpatialCol2ImTest, DilationSpacesTheTapsAWindowFolds)
        {
            // Two taps two columns apart over three pixels read pixels 0 and 2. See the test README.
            Spatial::ConvGeometry g;
            g.kernel_w = 2;
            g.dilation_w = 2;
            const int64_t out_w = Spatial::output_extent(3, 0, 0, 2, 2, 1);
            ASSERT_EQ(out_w, int64_t{1});

            Tensor cols = filled_cols(1, 2, 1.0f);
            Tensor dx = engine->create_tensor({1, 1, 3, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).col2im(cols, dx, g, 1, out_w);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, {1.0f, 0.0f, 1.0f});
        }

        TEST_F(Small_SpatialCol2ImTest, PixelsNoWindowReadsAreWrittenZeroRatherThanLeftAlone)
        {
            // Pixels no window reads must come back zero. See tests/operations/spatial/README.md.
            const Spatial::ConvGeometry g = two_by_two_stride_two();
            const int64_t H = 5, W = 5;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 2);
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 2);

            Tensor cols = filled_cols(out_h * out_w, 4, 1.0f);
            Tensor dx = engine->create_tensor({1, H, W, 1}, Core::DataType::Float32);
            fill_tensor(dx, std::vector<float>(25, -12345.0f));

            SushiBLAS::SpatialOps(recorder()).col2im(cols, dx, g, out_h, out_w);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, {1, 1, 1, 1, 0,
                                     1, 1, 1, 1, 0,
                                     1, 1, 1, 1, 0,
                                     1, 1, 1, 1, 0,
                                     0, 0, 0, 0, 0});
        }

        TEST_F(Medium_SpatialCol2ImTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 1;
            g.dilation_h = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t N = 2, H = 7, W = 5, C = 3;
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 2, 3, 2);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 2, 1);

            Tensor cols = engine->create_tensor({N * out_h * out_w, 3 * 2 * C},
                                                Core::DataType::Float32);
            fill_tensor(cols, cycling_values(cols.num_elements, 17, 8.0f));

            Tensor from_reference = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor from_device = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).col2im(cols, from_reference, g, out_h, out_w);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).col2im(cols, from_device, g, out_h, out_w);
            recording->run().value();

            expect_tensor_equal(from_device, read_tensor<float>(from_reference));
        }
    } // namespace Test
} // namespace SushiBLAS
