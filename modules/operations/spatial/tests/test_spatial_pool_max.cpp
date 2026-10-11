/**************************************************************************/
/* test_spatial_pool_max.cpp                                              */
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
 * @file test_spatial_pool_max.cpp
 * @brief Tests SpatialOps::pool_max and pool_max_backward.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <initializer_list>
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
            uint32_t pool_max_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialPoolMaxTest : public SushiBLASTest
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

                /**
                 * @brief Returns the 1x4x4x1 image the hand-checked cases pool.
                 * @see tests/unit/README.md
                 */
                Tensor four_by_four()
                {
                    return image(1, 4, 4, 1, {1, 3, 2, 0,
                                              4, 2, 1, 5,
                                              0, 7, 6, 2,
                                              8, 1, 3, 9});
                }

                /** @brief A 2x2 stride-2 geometry. */
                static Spatial::ConvGeometry two_by_two_stride_two()
                {
                    Spatial::ConvGeometry g;
                    g.kernel_h = 2;
                    g.kernel_w = 2;
                    g.stride_h = 2;
                    g.stride_w = 2;
                    return g;
                }
        };

        /**
         * @brief Tests that a padded tap never wins a window when every element is negative.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_SpatialPoolMaxTest, PaddingNeverWinsAWindow)
        {
            // Window (0,0) has one tap inside the image. See tests/unit/README.md.
            auto in = image(1, 2, 2, 1, {-4, -3,
                                         -2, -1});

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            // Padding above and left only puts the first window outside the top-left corner.
            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, g);
            engine->execute().value();

            const float* got = out.data_as<float>();
            EXPECT_FLOAT_EQ(got[0], -4.0f) << "the top-left window saw only element (0,0)";
            EXPECT_FLOAT_EQ(got[1], -3.0f) << "saw (0,0) and (0,1)";
            EXPECT_FLOAT_EQ(got[2], -2.0f) << "saw (0,0) and (1,0)";
            EXPECT_FLOAT_EQ(got[3], -1.0f) << "saw the whole image";
        }

        TEST_F(Unit_SpatialPoolMaxTest, ThePoolTakesTheWindowMaximum)
        {
            auto in = four_by_four();
            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, two_by_two_stride_two());
            engine->execute().value();

            const float* got = out.data_as<float>();
            EXPECT_FLOAT_EQ(got[0], 4.0f);   // max{1,3,4,2}
            EXPECT_FLOAT_EQ(got[1], 5.0f);   // max{2,0,1,5}
            EXPECT_FLOAT_EQ(got[2], 8.0f);   // max{0,7,8,1}
            EXPECT_FLOAT_EQ(got[3], 9.0f);   // max{6,2,3,9}
        }

        TEST_F(Unit_SpatialPoolMaxTest, TheIndexNamesTheElementTheMaximumCameFrom)
        {
            // The index is the logical flat NHWC position. See tests/unit/README.md.
            auto in = four_by_four();
            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, two_by_two_stride_two());
            engine->execute().value();

            const int64_t* got = indices.data_as<int64_t>();
            EXPECT_EQ(got[0], 4);    // value 4 sits at (h=1, w=0)
            EXPECT_EQ(got[1], 7);    // value 5 sits at (h=1, w=3)
            EXPECT_EQ(got[2], 12);   // value 8 sits at (h=3, w=0)
            EXPECT_EQ(got[3], 15);   // value 9 sits at (h=3, w=3)
        }

        TEST_F(Unit_SpatialPoolMaxTest, ATieGoesToTheFirstElementInScanOrder)
        {
            // A tie goes to the first tap in scan order. See tests/unit/README.md.
            auto in = image(1, 2, 2, 1, {5, 5,
                                         5, 5});
            auto out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, two_by_two_stride_two());
            engine->execute().value();

            EXPECT_FLOAT_EQ(out.data_as<float>()[0], 5.0f);
            EXPECT_EQ(indices.data_as<int64_t>()[0], 0) << "the window's first tap, in (r, s) order";
        }

        TEST_F(Unit_SpatialPoolMaxTest, TheBackwardRoutesEachGradientToItsArgmaxAndNowhereElse)
        {
            auto in = four_by_four();
            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            const Spatial::ConvGeometry g = two_by_two_stride_two();
            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, g);
            engine->execute().value();

            auto dy = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            float* d = dy.data_as<float>();
            d[0] = 10.0f; d[1] = 20.0f; d[2] = 30.0f; d[3] = 40.0f;

            auto dx = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(*engine).pool_max_backward(dy, indices, dx, g);
            engine->execute().value();

            // Every non-argmax pixel is exactly zero. See tests/unit/README.md.
            const std::vector<float> expected = {0, 0, 0, 0,
                                                 10, 0, 0, 20,
                                                 0, 0, 0, 0,
                                                 30, 0, 0, 40};

            const float* got = dx.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "pixel " << i;
        }

        TEST_F(Unit_SpatialPoolMaxTest, OverlappingWindowsAccumulateAtASharedArgmax)
        {
            // All four windows select the one dominant pixel. See tests/unit/README.md.
            auto in = image(1, 3, 3, 1, {0, 0, 0,
                                         0, 9, 0,
                                         0, 0, 0});

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);
            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, g);
            engine->execute().value();

            const int64_t* selected = indices.data_as<int64_t>();
            for (int64_t k = 0; k < 4; ++k)
                ASSERT_EQ(selected[k], 4) << "window " << k << " must select the centre pixel";

            auto dy = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            float* d = dy.data_as<float>();
            d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f;

            auto dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(*engine).pool_max_backward(dy, indices, dx, g);
            engine->execute().value();

            const float* got = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], i == 4 ? 10.0f : 0.0f) << "pixel " << i;
        }

        TEST_F(Unit_SpatialPoolMaxTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/unit/README.md.
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

            auto out_ref = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            auto idx_ref = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);
            auto out_dev = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            auto idx_dev = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);

            engine->set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(*engine).pool_max(in, out_ref, idx_ref, g);
            engine->execute().value();

            engine->set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(*engine).pool_max(in, out_dev, idx_dev, g);
            engine->execute().value();

            const float* value_expected = out_ref.data_as<float>();
            const float* value_got = out_dev.data_as<float>();
            const int64_t* index_expected = idx_ref.data_as<int64_t>();
            const int64_t* index_got = idx_dev.data_as<int64_t>();

            for (int64_t i = 0; i < out_ref.num_elements; ++i)
            {
                EXPECT_EQ(pool_max_float_bits(value_got[i]), pool_max_float_bits(value_expected[i]))
                    << "value " << i;
                EXPECT_EQ(index_got[i], index_expected[i]) << "index " << i;
            }
        }
    }
}
