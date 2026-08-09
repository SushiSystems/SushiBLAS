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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <vector>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The bit pattern of a float; see test_spatial_im2col.cpp. */
            uint32_t col2im_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialCol2ImTest : public SushiBLASTest
        {
            protected:
                /** @brief A rank-2 column matrix whose every entry is @p value. */
                Tensor filled_cols(int64_t rows, int64_t width, float value)
                {
                    auto t = engine->create_tensor({rows, width}, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i)
                        p[i] = value;
                    return t;
                }
        };

        TEST_F(Unit_SpatialCol2ImTest, EachPixelReceivesOncePerWindowThatReadIt)
        {
            // col2im of an all-ones column matrix is the *overlap count* of each
            // pixel: how many windows read it. That is a closed-form answer this
            // test computes independently, and it is the strongest cheap check
            // that the gather visits exactly the right set - a rule that drops
            // contributions or double-counts fails it, where a round-trip
            // against im2col would not necessarily.
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t N = 2, H = 4, W = 5, C = 2;
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);

            auto cols = filled_cols(N * out_h * out_w, 3 * 3 * C, 1.0f);
            auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);

            engine->spatial().col2im(cols, dx, g, out_h, out_w);
            engine->execute();

            const float* got = dx.data_as<float>();
            for (int64_t n = 0; n < N; ++n)
                for (int64_t h = 0; h < H; ++h)
                    for (int64_t w = 0; w < W; ++w)
                    {
                        // With a 3x3 stride-1 window padded by 1, a pixel is read
                        // by every window whose origin is within one step of it and
                        // inside the image: 3x3 in the interior, fewer at the edges.
                        // Counted by brute force rather than by a closed form, so
                        // this expectation cannot share a mistake with the kernel.
                        int64_t count = 0;
                        for (int64_t ho = 0; ho < out_h; ++ho)
                            for (int64_t wo = 0; wo < out_w; ++wo)
                                for (int64_t r = 0; r < 3; ++r)
                                    for (int64_t s = 0; s < 3; ++s)
                                        if (ho - 1 + r == h && wo - 1 + s == w) ++count;

                        for (int64_t ch = 0; ch < C; ++ch)
                            EXPECT_FLOAT_EQ(got[((n * H + h) * W + w) * C + ch],
                                            static_cast<float>(count))
                                << "pixel (" << n << "," << h << "," << w << "," << ch << ")";
                    }
        }

        TEST_F(Unit_SpatialCol2ImTest, StrideAndDilationSelectTheRightContributors)
        {
            // A stride-2 window means every pixel here is read by exactly one
            // window. Getting the divisibility test wrong - dropping the
            // `% stride != 0` guard - turns this into a count of 4 and is
            // otherwise invisible.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            const int64_t H = 4, W = 4, C = 1;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 2);   // 2
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 2);   // 2

            auto cols = filled_cols(out_h * out_w, 2 * 2 * C, 1.0f);
            auto dx = engine->create_tensor({1, H, W, C}, Core::DataType::FLOAT32);

            engine->spatial().col2im(cols, dx, g, out_h, out_w);
            engine->execute();

            // Non-overlapping 2x2 tiles: every pixel is read exactly once.
            const float* got = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], 1.0f) << "element " << i;
        }

        TEST_F(Unit_SpatialCol2ImTest, PixelsNoWindowReadsAreWrittenZeroRatherThanLeftAlone)
        {
            // A 2x2 stride-2 window over a 5x5 image covers rows 0..3 and
            // columns 0..3; the last row and column are read by nothing. The
            // gather form writes every pixel exactly once, so those must come
            // back zero - which is what lets a caller pass an uninitialised
            // destination. Seeded with a sentinel so "left alone" is visible.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            const int64_t H = 5, W = 5, C = 1;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 2);   // 2
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 2);   // 2

            auto cols = filled_cols(out_h * out_w, 2 * 2 * C, 1.0f);
            auto dx = engine->create_tensor({1, H, W, C}, Core::DataType::FLOAT32);

            const float sentinel = -12345.0f;
            float* seed = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                seed[i] = sentinel;

            engine->spatial().col2im(cols, dx, g, out_h, out_w);
            engine->execute();

            const float* got = dx.data_as<float>();
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w)
                {
                    const bool covered = h < 4 && w < 4;
                    EXPECT_FLOAT_EQ(got[h * W + w], covered ? 1.0f : 0.0f)
                        << "pixel (" << h << "," << w << ")";
                }
        }

        TEST_F(Unit_SpatialCol2ImTest, TheSyclKernelAgreesWithTheReferenceKernel)
        {
            // Compared here rather than under ExecutionMode::VERIFY, which runs
            // both kernels but lets the reference overwrite the device's answer
            // instead of checking it (see Internal::dispatch_with_mode).
            //
            // The column entries are a ramp rather than all ones, so a pixel's
            // total depends on *which* entries it gathered and not only on how
            // many - the all-ones case cannot tell a wrong-but-equal-sized set
            // from the right one.
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

            auto cols = engine->create_tensor({N * out_h * out_w, 3 * 2 * C},
                                              Core::DataType::FLOAT32);
            float* p = cols.data_as<float>();
            for (int64_t i = 0; i < cols.num_elements; ++i)
                p[i] = static_cast<float>(i % 17) - 8.0f;

            auto from_reference = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto from_device = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);

            engine->set_mode(ExecutionMode::FORCE_REFERENCE);
            engine->spatial().col2im(cols, from_reference, g, out_h, out_w);
            engine->execute();

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            engine->spatial().col2im(cols, from_device, g, out_h, out_w);
            engine->execute();

            const float* expected = from_reference.data_as<float>();
            const float* got = from_device.data_as<float>();
            for (int64_t i = 0; i < from_reference.num_elements; ++i)
                EXPECT_EQ(col2im_float_bits(got[i]), col2im_float_bits(expected[i]))
                    << "element " << i << ": reference " << expected[i]
                    << ", device " << got[i];
        }
    }
}
