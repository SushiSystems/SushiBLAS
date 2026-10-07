/**************************************************************************/
/* test_spatial_broadcast.cpp                                             */
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
 * @file test_spatial_broadcast.cpp
 * @brief Tests SpatialOps::channel_affine, broadcast_spatial and layout_convert.
 * @author Mustafa Garip
 */

#include <array>
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
            /** @brief The bit pattern of a float; see test_spatial_pool_avg.cpp. */
            uint32_t broadcast_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialBroadcastTest : public SushiBLASTest
        {
            protected:
                /** @brief A float tensor of the given shape holding @p values in memory order. */
                Tensor filled(std::initializer_list<int64_t> dims,
                              std::initializer_list<float> values)
                {
                    auto t = engine->create_tensor(dims, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    int64_t i = 0;
                    for (float v : values) p[i++] = v;
                    return t;
                }

                /** @brief A float tensor of the given shape whose element k holds `k`. */
                Tensor ramp(std::initializer_list<int64_t> dims)
                {
                    auto t = engine->create_tensor(dims, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i) p[i] = static_cast<float>(i);
                    return t;
                }

                /** @brief A float tensor of the given shape whose every element is @p value. */
                Tensor uniform(std::initializer_list<int64_t> dims, float value)
                {
                    auto t = engine->create_tensor(dims, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i) p[i] = value;
                    return t;
                }

                /**
                 * @brief Returns an NCHW-strided view of @p base, logically [N, H, W, C].
                 * @param base Contiguous, at least N*H*W*C elements; the view borrows its storage.
                 * @return The view, whose channel stride is h*w and whose column stride is 1.
                 * @see tests/functional/unit/README.md
                 */
                Tensor nchw_view(const Tensor& base, int64_t n, int64_t h, int64_t w, int64_t c)
                {
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> s{};
                    s[0] = c * h * w;   // N
                    s[1] = w;           // H
                    s[2] = 1;           // W
                    s[3] = h * w;       // C

                    const int64_t dims[4] = {n, h, w, c};
                    return Tensor(base.storage,
                                  SushiRuntime::span<const int64_t>(dims, dims + 4), s,
                                  Core::DataType::Float32, base.storage_offset, base.layout);
                }
        };

        TEST_F(Unit_SpatialBroadcastTest, ChannelAffineAppliesThePerChannelPair)
        {
            auto in = ramp({1, 2, 2, 3});
            auto gamma = filled({3}, {2.0f, 3.0f, 4.0f});
            auto beta = filled({3}, {10.0f, 20.0f, 30.0f});
            auto out = engine->create_tensor({1, 2, 2, 3}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).channel_affine(in, gamma, beta, out);
            engine->execute();

            // Element k has channel k % 3, so it becomes k * gamma[k % 3] + beta[k % 3].
            const std::vector<float> expected = {10, 23, 38,
                                                 16, 32, 50,
                                                 22, 41, 62,
                                                 28, 50, 74};

            const float* got = out.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "element " << i;
        }

        TEST_F(Unit_SpatialBroadcastTest, ChannelAffineWithUnitGammaAndZeroBetaIsACopy)
        {
            auto in = ramp({2, 3, 4, 5});
            auto gamma = uniform({5}, 1.0f);
            auto beta = uniform({5}, 0.0f);
            auto out = engine->create_tensor({2, 3, 4, 5}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).channel_affine(in, gamma, beta, out);
            engine->execute();

            const float* source = in.data_as<float>();
            const float* got = out.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                EXPECT_EQ(broadcast_float_bits(got[i]), broadcast_float_bits(source[i]))
                    << "element " << i;
        }

        TEST_F(Unit_SpatialBroadcastTest, BroadcastSpatialSpreadsAndScales)
        {
            const int64_t N = 2, H = 4, W = 5, C = 3;

            auto in = filled({N, C}, {1, 2, 3,
                                      4, 5, 6});
            auto out = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).broadcast_spatial(in, out, 0.5);
            engine->execute();

            const float* source = in.data_as<float>();
            const float* got = out.data_as<float>();
            for (int64_t i = 0; i < out.num_elements; ++i)
            {
                const int64_t c = i % C;
                const int64_t n = i / (H * W * C);
                EXPECT_FLOAT_EQ(got[i], source[n * C + c] * 0.5f)
                    << "element " << i << " (n " << n << ", c " << c << ")";
            }
        }

        TEST_F(Unit_SpatialBroadcastTest, BroadcastSpatialAtOneOverHwIsTheMeanBackward)
        {
            const int64_t N = 2, H = 4, W = 5, C = 3;
            const double scale = 1.0 / static_cast<double>(H * W);

            auto in = uniform({N, C}, 1.0f);
            auto out = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(*engine).broadcast_spatial(in, out, scale);
            engine->execute();

            // float(1/20) is one rounding of an exact double, so every element is identical.
            const float share = static_cast<float>(scale);
            const float* got = out.data_as<float>();
            for (int64_t i = 0; i < out.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], share) << "element " << i;

            // Each (n, c) plane sums back to 1.0 within 1e-6. See tests/functional/unit/README.md.
            for (int64_t n = 0; n < N; ++n)
            {
                for (int64_t c = 0; c < C; ++c)
                {
                    float total = 0.0f;
                    for (int64_t h = 0; h < H; ++h)
                    {
                        for (int64_t w = 0; w < W; ++w)
                            total += got[((n * H + h) * W + w) * C + c];
                    }
                    EXPECT_NEAR(total, 1.0f, 1e-6f) << "plane (n " << n << ", c " << c << ")";
                }
            }
        }

        TEST_F(Unit_SpatialBroadcastTest, LayoutConvertRoundTripsToItself)
        {
            const int64_t N = 2, H = 3, W = 4, C = 5;

            auto source = ramp({N, H, W, C});
            auto middle_storage = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto middle = nchw_view(middle_storage, N, H, W, C);
            auto destination = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            // NHWC to NCHW and back: the same call, with the permuted strides on the other operand.
            SushiBLAS::SpatialOps(*engine).layout_convert(source, middle);
            engine->execute();
            SushiBLAS::SpatialOps(*engine).layout_convert(middle, destination);
            engine->execute();

            const float* expected = source.data_as<float>();
            const float* got = destination.data_as<float>();
            for (int64_t i = 0; i < source.num_elements; ++i)
                EXPECT_EQ(broadcast_float_bits(got[i]), broadcast_float_bits(expected[i]))
                    << "element " << i;
        }

        TEST_F(Unit_SpatialBroadcastTest, LayoutConvertMovesTheChannelIndex)
        {
            // A round trip cannot catch a no-op conversion, so the intermediate buffer is read.
            const int64_t N = 1, H = 2, W = 2, C = 2;

            auto source = ramp({N, H, W, C});
            auto nchw_storage = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto nchw = nchw_view(nchw_storage, N, H, W, C);

            SushiBLAS::SpatialOps(*engine).layout_convert(source, nchw);
            engine->execute();

            // Element (h, w, c) sits at h*4 + w*2 + c and must land at c*4 + h*2 + w in NCHW.
            const std::vector<float> expected = {0, 2, 4, 6,    // channel 0
                                                 1, 3, 5, 7};   // channel 1

            const float* got = nchw_storage.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "NCHW element " << i;
        }

        TEST_F(Unit_SpatialBroadcastTest, TheSyclKernelsAgreeWithTheReferenceKernels)
        {
            // The two forced modes are compared here. See tests/functional/unit/README.md.
            const int64_t N = 2, H = 5, W = 3, C = 4;

            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                p[i] = static_cast<float>((i * 37) % 23) - 11.0f;

            auto gamma = engine->create_tensor({C}, Core::DataType::Float32);
            auto beta = engine->create_tensor({C}, Core::DataType::Float32);
            float* g = gamma.data_as<float>();
            float* b = beta.data_as<float>();
            for (int64_t c = 0; c < C; ++c)
            {
                g[c] = 0.5f + static_cast<float>(c) * 0.25f;
                b[c] = static_cast<float>(c) - 1.5f;
            }

            auto plane = engine->create_tensor({N, C}, Core::DataType::Float32);
            float* pl = plane.data_as<float>();
            for (int64_t i = 0; i < plane.num_elements; ++i)
                pl[i] = static_cast<float>((i * 13) % 7) - 3.0f;

            auto affine_reference = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto affine_device = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto spread_reference = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto spread_device = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto convert_reference_storage =
                engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto convert_device_storage =
                engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            auto convert_reference = nchw_view(convert_reference_storage, N, H, W, C);
            auto convert_device = nchw_view(convert_device_storage, N, H, W, C);

            engine->set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(*engine).channel_affine(in, gamma, beta, affine_reference);
            SushiBLAS::SpatialOps(*engine).broadcast_spatial(plane, spread_reference, 1.0 / 15.0);
            SushiBLAS::SpatialOps(*engine).layout_convert(in, convert_reference);
            engine->execute();

            engine->set_mode(ExecutionMode::ForceSycl);
            SushiBLAS::SpatialOps(*engine).channel_affine(in, gamma, beta, affine_device);
            SushiBLAS::SpatialOps(*engine).broadcast_spatial(plane, spread_device, 1.0 / 15.0);
            SushiBLAS::SpatialOps(*engine).layout_convert(in, convert_device);
            engine->execute();

            const float* affine_expected = affine_reference.data_as<float>();
            const float* affine_got = affine_device.data_as<float>();
            const float* spread_expected = spread_reference.data_as<float>();
            const float* spread_got = spread_device.data_as<float>();
            const float* convert_expected = convert_reference_storage.data_as<float>();
            const float* convert_got = convert_device_storage.data_as<float>();

            for (int64_t i = 0; i < in.num_elements; ++i)
            {
                EXPECT_EQ(broadcast_float_bits(affine_got[i]),
                          broadcast_float_bits(affine_expected[i]))
                    << "channel_affine element " << i;
                EXPECT_EQ(broadcast_float_bits(spread_got[i]),
                          broadcast_float_bits(spread_expected[i]))
                    << "broadcast_spatial element " << i;
                EXPECT_EQ(broadcast_float_bits(convert_got[i]),
                          broadcast_float_bits(convert_expected[i]))
                    << "layout_convert element " << i;
            }
        }
    }
}
