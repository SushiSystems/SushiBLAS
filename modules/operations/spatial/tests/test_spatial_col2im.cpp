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
 * @file test_spatial_col2im.cpp
 * @brief Tests SpatialOps::col2im against overlap counts and the reference kernel.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

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
                    auto t = engine->create_tensor({rows, width}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i)
                        p[i] = value;
                    return t;
                }
        };

        TEST_F(Unit_SpatialCol2ImTest, EachPixelReceivesOncePerWindowThatReadIt)
        {
            // All-ones columns give a pixel's overlap count. See tests/unit/README.md.
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t N = 2, H = 4, W = 5, C = 2;
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);

            auto cols = filled_cols(N * out_h * out_w, 3 * 3 * C, 1.0f);
            auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).col2im(cols, dx, g, out_h, out_w);
            engine->execute().value();

            const float* got = dx.data_as<float>();
            for (int64_t n = 0; n < N; ++n)
                for (int64_t h = 0; h < H; ++h)
                    for (int64_t w = 0; w < W; ++w)
                    {
                        // Windows are counted by brute force. See tests/unit/README.md.
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
            // With stride 2 each pixel has exactly one window. See tests/unit/README.md.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            const int64_t H = 4, W = 4, C = 1;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 2);   // 2
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 2);   // 2

            auto cols = filled_cols(out_h * out_w, 2 * 2 * C, 1.0f);
            auto dx = engine->create_tensor({1, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).col2im(cols, dx, g, out_h, out_w);
            engine->execute().value();

            // Non-overlapping 2x2 tiles: every pixel is read exactly once.
            const float* got = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], 1.0f) << "element " << i;
        }

        TEST_F(Unit_SpatialCol2ImTest, PixelsNoWindowReadsAreWrittenZeroRatherThanLeftAlone)
        {
            // Pixels no window reads must come back zero. See tests/unit/README.md.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            const int64_t H = 5, W = 5, C = 1;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 2);   // 2
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 2);   // 2

            auto cols = filled_cols(out_h * out_w, 2 * 2 * C, 1.0f);
            auto dx = engine->create_tensor({1, H, W, C}, Core::DataType::Float32);

            const float sentinel = -12345.0f;
            float* seed = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                seed[i] = sentinel;

            SushiBLAS::SpatialOps(*engine).col2im(cols, dx, g, out_h, out_w);
            engine->execute().value();

            const float* got = dx.data_as<float>();
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w)
                {
                    const bool covered = h < 4 && w < 4;
                    EXPECT_FLOAT_EQ(got[h * W + w], covered ? 1.0f : 0.0f)
                        << "pixel (" << h << "," << w << ")";
                }
        }

        TEST_F(Unit_SpatialCol2ImTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/unit/README.md.
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
                                              Core::DataType::Float32);
            float* p = cols.data_as<float>();
            for (int64_t i = 0; i < cols.num_elements; ++i)
                p[i] = static_cast<float>(i % 17) - 8.0f;

            auto from_reference = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto from_device = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            engine->set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(*engine).col2im(cols, from_reference, g, out_h, out_w);
            engine->execute().value();

            engine->set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(*engine).col2im(cols, from_device, g, out_h, out_w);
            engine->execute().value();

            const float* expected = from_reference.data_as<float>();
            const float* got = from_device.data_as<float>();
            for (int64_t i = 0; i < from_reference.num_elements; ++i)
                EXPECT_EQ(col2im_float_bits(got[i]), col2im_float_bits(expected[i]))
                    << "element " << i << ": reference " << expected[i]
                    << ", device " << got[i];
        }
    }
}
