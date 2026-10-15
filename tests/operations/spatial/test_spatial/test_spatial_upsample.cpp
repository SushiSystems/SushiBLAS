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
 * @file test_spatial_upsample.cpp
 * @brief Tests SpatialOps::upsample_nearest and upsample_nearest_backward.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The bit pattern of a float; see test_spatial_pool_avg.cpp. */
            uint32_t upsample_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialUpsampleTest : public SushiBLASTest
        {
            protected:
                /** @brief A rank-4 NHWC float tensor holding @p values in memory order. */
                Tensor image(int64_t n, int64_t h, int64_t w, int64_t c,
                             std::initializer_list<float> values)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    int64_t i = 0;
                    for (float v : values)
                        p[i++] = v;
                    return t;
                }

                /** @brief A rank-4 NHWC float tensor holding 0, 1, 2, ... in memory order. */
                Tensor ramp(int64_t n, int64_t h, int64_t w, int64_t c)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i)
                        p[i] = static_cast<float>(i);
                    return t;
                }
        };

        TEST_F(Unit_SpatialUpsampleTest, EachInputPixelBecomesAScaleByScaleBlock)
        {
            auto in = image(1, 2, 2, 1, {0, 1,
                                         2, 3});

            auto out = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(*engine).upsample_nearest(in, out, 2, 2);
            engine->execute().value();

            const std::vector<float> expected = {0, 0, 1, 1,
                                                 0, 0, 1, 1,
                                                 2, 2, 3, 3,
                                                 2, 2, 3, 3};

            const float* got = out.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "pixel " << i;
        }

        TEST_F(Unit_SpatialUpsampleTest, AnisotropicScalesAreIndependent)
        {
            // Each pixel becomes a 2-row by 3-column block, so the 2x2 input becomes 4x6.
            auto in = image(1, 2, 2, 1, {1, 2,
                                         3, 4});

            auto out = engine->create_tensor({1, 4, 6, 1}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(*engine).upsample_nearest(in, out, 2, 3);
            engine->execute().value();

            const std::vector<float> expected = {1, 1, 1, 2, 2, 2,
                                                 1, 1, 1, 2, 2, 2,
                                                 3, 3, 3, 4, 4, 4,
                                                 3, 3, 3, 4, 4, 4};

            const float* got = out.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "pixel " << i;
        }

        TEST_F(Unit_SpatialUpsampleTest, AUnitScaleIsACopy)
        {
            auto in = ramp(2, 3, 4, 2);
            auto out = engine->create_tensor({2, 3, 4, 2}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).upsample_nearest(in, out, 1, 1);
            engine->execute().value();

            const float* expected = in.data_as<float>();
            const float* got = out.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "element " << i;
        }

        TEST_F(Unit_SpatialUpsampleTest, TheBackwardSumsTheBlockThatReadEachInput)
        {
            const int64_t N = 2, H = 3, W = 2, C = 2;
            const int64_t scale_h = 2, scale_w = 3;
            const int64_t out_h = H * scale_h, out_w = W * scale_w;

            auto dy = ramp(N, out_h, out_w, C);
            auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).upsample_nearest_backward(dy, dx, scale_h, scale_w);
            engine->execute().value();

            // The block is summed index by index. See tests/operations/spatial/README.md.
            const float* source = dy.data_as<float>();
            const float* got = dx.data_as<float>();

            for (int64_t n = 0; n < N; ++n)
            {
                for (int64_t h = 0; h < H; ++h)
                {
                    for (int64_t w = 0; w < W; ++w)
                    {
                        for (int64_t c = 0; c < C; ++c)
                        {
                            float expected = 0.0f;

                            for (int64_t dh = 0; dh < scale_h; ++dh)
                            {
                                for (int64_t dw = 0; dw < scale_w; ++dw)
                                {
                                    const int64_t row = h * scale_h + dh;
                                    const int64_t col = w * scale_w + dw;
                                    expected += source[((n * out_h + row) * out_w + col) * C + c];
                                }
                            }

                            const int64_t i = ((n * H + h) * W + w) * C + c;
                            EXPECT_FLOAT_EQ(got[i], expected) << "element " << i;
                        }
                    }
                }
            }
        }

        TEST_F(Unit_SpatialUpsampleTest, AScaleOfZeroOrNegativeIsRefused)
        {
            auto in = image(1, 2, 2, 1, {1, 2,
                                         3, 4});
            auto out = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            bool zero_threw = false;
            try
            {
                SushiBLAS::SpatialOps(*engine).upsample_nearest(in, out, 0, 2);
            }
            catch (const std::exception& e)
            {
                zero_threw = true;
                const std::string message = e.what();
                EXPECT_NE(message.find("spatial.upsample_nearest"), std::string::npos)
                    << "message did not name the operation: " << message;
            }
            EXPECT_TRUE(zero_threw) << "a zero scale was accepted";

            bool negative_threw = false;
            try
            {
                SushiBLAS::SpatialOps(*engine).upsample_nearest(in, out, 2, -3);
            }
            catch (const std::exception& e)
            {
                negative_threw = true;
                const std::string message = e.what();
                EXPECT_NE(message.find("spatial.upsample_nearest"), std::string::npos)
                    << "message did not name the operation: " << message;
            }
            EXPECT_TRUE(negative_threw) << "a negative scale was accepted";
        }

        TEST_F(Unit_SpatialUpsampleTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            const int64_t N = 2, H = 5, W = 3, C = 3;
            const int64_t scale_h = 3, scale_w = 2;
            const int64_t out_h = H * scale_h, out_w = W * scale_w;

            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                p[i] = static_cast<float>((i * 37) % 23) - 11.0f;

            auto from_reference = engine->create_tensor({N, out_h, out_w, C},
                                                        Core::DataType::Float32);
            auto from_device = engine->create_tensor({N, out_h, out_w, C},
                                                     Core::DataType::Float32);

            engine->set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(*engine).upsample_nearest(in, from_reference, scale_h, scale_w);
            engine->execute().value();

            engine->set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(*engine).upsample_nearest(in, from_device, scale_h, scale_w);
            engine->execute().value();

            const float* expected = from_reference.data_as<float>();
            const float* got = from_device.data_as<float>();
            for (int64_t i = 0; i < from_reference.num_elements; ++i)
                EXPECT_EQ(upsample_float_bits(got[i]), upsample_float_bits(expected[i]))
                    << "element " << i << ": reference " << expected[i]
                    << ", device " << got[i];
        }
    }
}
