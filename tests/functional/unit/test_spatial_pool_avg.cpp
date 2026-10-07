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
 * @file test_spatial_pool_avg.cpp
 * @brief Tests SpatialOps::pool_avg and pool_avg_backward.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <initializer_list>
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
            uint32_t pool_avg_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialPoolAvgTest : public SushiBLASTest
        {
            protected:
                /** @brief A rank-4 NHWC float tensor holding @p values in memory order. */
                Tensor image(int64_t n, int64_t h, int64_t w, int64_t c,
                             std::initializer_list<float> values)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    int64_t i = 0;
                    for (float v : values) p[i++] = v;
                    return t;
                }

                /** @brief A rank-4 NHWC float tensor whose every element is @p value. */
                Tensor uniform(int64_t n, int64_t h, int64_t w, int64_t c, float value)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i) p[i] = value;
                    return t;
                }
        };

        /**
         * @brief Tests that the average divides by the whole window area, padded taps included.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_SpatialPoolAvgTest, TheAverageIsOverTheWholeWindowIncludingPaddedTaps)
        {
            auto in = image(1, 2, 2, 1, {4, 8,
                                         12, 16});

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(*engine).pool_avg(in, out, g);
            engine->execute();

            const float* got = out.data_as<float>();

            // The window's only in-bounds tap is element 4. See tests/functional/unit/README.md.
            EXPECT_FLOAT_EQ(got[0], 1.0f) << "one real tap out of four, so 4/4";
            EXPECT_FLOAT_EQ(got[1], 3.0f) << "(4 + 8) / 4";
            EXPECT_FLOAT_EQ(got[2], 4.0f) << "(4 + 12) / 4";
            EXPECT_FLOAT_EQ(got[3], 10.0f) << "(4 + 8 + 12 + 16) / 4";
        }

        TEST_F(Unit_SpatialPoolAvgTest, AnUnpaddedPoolIsThePlainMean)
        {
            auto in = image(1, 4, 4, 1, {1, 3, 2, 0,
                                         4, 2, 1, 5,
                                         0, 7, 6, 2,
                                         8, 1, 3, 9});

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(*engine).pool_avg(in, out, g);
            engine->execute();

            const float* got = out.data_as<float>();
            EXPECT_FLOAT_EQ(got[0], 2.5f);   // (1+3+4+2)/4
            EXPECT_FLOAT_EQ(got[1], 2.0f);   // (2+0+1+5)/4
            EXPECT_FLOAT_EQ(got[2], 4.0f);   // (0+7+8+1)/4
            EXPECT_FLOAT_EQ(got[3], 5.0f);   // (6+2+3+9)/4
        }

        TEST_F(Unit_SpatialPoolAvgTest, TheBackwardSpreadsEachGradientOverItsWholeWindow)
        {
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            auto dy = image(1, 2, 2, 1, {4, 8,
                                         12, 16});
            auto dx = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).pool_avg_backward(dy, dx, g);
            engine->execute();

            // Non-overlapping quadrants, each receiving its window's dy/4.
            const std::vector<float> expected = {1, 1, 2, 2,
                                                 1, 1, 2, 2,
                                                 3, 3, 4, 4,
                                                 3, 3, 4, 4};

            const float* got = dx.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "pixel " << i;
        }

        TEST_F(Unit_SpatialPoolAvgTest, OverlappingWindowsSumTheirShares)
        {
            // The centre pixel is covered by all four windows. See tests/functional/unit/README.md.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            auto dy = image(1, 2, 2, 1, {1, 2,
                                         3, 4});
            auto dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).pool_avg_backward(dy, dx, g);
            engine->execute();

            const float* got = dx.data_as<float>();
            EXPECT_FLOAT_EQ(got[0], 0.25f);                    // (0,0): window 0 only
            EXPECT_FLOAT_EQ(got[2], 0.5f);                     // (0,2): window 1 only
            EXPECT_FLOAT_EQ(got[4], (1 + 2 + 3 + 4) / 4.0f);   // (1,1): all four
            EXPECT_FLOAT_EQ(got[6], 0.75f);                    // (2,0): window 2 only
            EXPECT_FLOAT_EQ(got[8], 1.0f);                     // (2,2): window 3 only
        }

        TEST_F(Unit_SpatialPoolAvgTest, TheBackwardOfAConstantForwardIsUniform)
        {
            // Every pixel is covered by exactly R*S windows. See tests/functional/unit/README.md.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            auto dy = uniform(1, 4, 4, 1, 1.0f);
            auto dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).pool_avg_backward(dy, dx, g);
            engine->execute();

            const float* got = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], 1.0f) << "pixel " << i;
        }

        TEST_F(Unit_SpatialPoolAvgTest, TheSyclKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/functional/unit/README.md.
            const int64_t N = 2, H = 7, W = 5, C = 3;

            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                p[i] = static_cast<float>((i * 37) % 23) - 11.0f;

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

            auto from_reference = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            auto from_device = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);

            engine->set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(*engine).pool_avg(in, from_reference, g);
            engine->execute();

            engine->set_mode(ExecutionMode::ForceSycl);
            SushiBLAS::SpatialOps(*engine).pool_avg(in, from_device, g);
            engine->execute();

            const float* expected = from_reference.data_as<float>();
            const float* got = from_device.data_as<float>();
            for (int64_t i = 0; i < from_reference.num_elements; ++i)
                EXPECT_EQ(pool_avg_float_bits(got[i]), pool_avg_float_bits(expected[i]))
                    << "element " << i << ": reference " << expected[i]
                    << ", device " << got[i];
        }
    }
}
