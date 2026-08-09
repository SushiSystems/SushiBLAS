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
                    auto t = engine->create_tensor(dims, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    int64_t i = 0;
                    for (float v : values) p[i++] = v;
                    return t;
                }

                /** @brief A float tensor of the given shape whose element k holds `k`. */
                Tensor ramp(std::initializer_list<int64_t> dims)
                {
                    auto t = engine->create_tensor(dims, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i) p[i] = static_cast<float>(i);
                    return t;
                }

                /** @brief A float tensor of the given shape whose every element is @p value. */
                Tensor uniform(std::initializer_list<int64_t> dims, float value)
                {
                    auto t = engine->create_tensor(dims, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i) p[i] = value;
                    return t;
                }

                /**
                 * @brief An NCHW-strided view of @p base's allocation, logically `[N, H, W, C]`.
                 *
                 * The public API does express a custom stride vector: `Tensor`'s
                 * stride-explicit constructor takes the whole fixed-width stride
                 * array, so nothing here reaches past the library's own seams.
                 * `create_tensor` only ever lays a shape out contiguously, so the
                 * allocation comes from it and the layout comes from here.
                 *
                 * The logical shape stays `[N, H, W, C]` because that is what
                 * `layout_convert` compares - a conversion permutes strides, not
                 * shapes.
                 *
                 * @param base A contiguous tensor of at least `N*H*W*C` elements,
                 *        whose storage this view borrows.
                 * @param n The batch count.
                 * @param h The row count.
                 * @param w The column count.
                 * @param c The channel count.
                 * @return The view, whose channel stride is `h*w` and whose column stride is 1.
                 */
                Tensor nchw_view(const Tensor& base, int64_t n, int64_t h, int64_t w, int64_t c)
                {
                    std::array<int64_t, Core::MAX_TENSOR_RANK> s{};
                    s[0] = c * h * w;   // N
                    s[1] = w;           // H
                    s[2] = 1;           // W
                    s[3] = h * w;       // C

                    const int64_t dims[4] = {n, h, w, c};
                    return Tensor(base.storage,
                                  SushiRuntime::span<const int64_t>(dims, dims + 4), s,
                                  Core::DataType::FLOAT32, base.storage_offset, base.layout);
                }
        };

        TEST_F(Unit_SpatialBroadcastTest, ChannelAffineAppliesThePerChannelPair)
        {
            auto in = ramp({1, 2, 2, 3});
            auto gamma = filled({3}, {2.0f, 3.0f, 4.0f});
            auto beta = filled({3}, {10.0f, 20.0f, 30.0f});
            auto out = engine->create_tensor({1, 2, 2, 3}, Core::DataType::FLOAT32);

            engine->spatial().channel_affine(in, gamma, beta, out);
            engine->execute();

            // Element k of the input is k, and its channel is k % 3, so the
            // expected value is k * gamma[k % 3] + beta[k % 3].
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
            auto out = engine->create_tensor({2, 3, 4, 5}, Core::DataType::FLOAT32);

            engine->spatial().channel_affine(in, gamma, beta, out);
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
            auto out = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);

            engine->spatial().broadcast_spatial(in, out, 0.5);
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
            auto out = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);

            engine->spatial().broadcast_spatial(in, out, scale);
            engine->execute();

            // Every element is 1 * float(1/20), which is a single rounding of an
            // exact double and therefore identical everywhere.
            const float share = static_cast<float>(scale);
            const float* got = out.data_as<float>();
            for (int64_t i = 0; i < out.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], share) << "element " << i;

            // The property SPATIAL_MEAN_BACKWARD needs: each (n, c) plane sums
            // back to the 1.0 that global average pooling's forward divided out.
            // Not exact, and not expected to be - 1/20 has no finite binary
            // expansion, so twenty copies of float(1/20) add up to 1.0 only to
            // within the rounding each addition carries. A few ulps of 1.0 is
            // roughly 1e-7; 1e-6 leaves room without hiding a real error, which
            // would be off by a factor of H, of W, or of N.
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
            auto middle_storage = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto middle = nchw_view(middle_storage, N, H, W, C);
            auto destination = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);

            // NHWC -> NCHW, then NCHW -> NHWC. The same call both times; only
            // which operand carries the permuted strides changes.
            engine->spatial().layout_convert(source, middle);
            engine->execute();
            engine->spatial().layout_convert(middle, destination);
            engine->execute();

            const float* expected = source.data_as<float>();
            const float* got = destination.data_as<float>();
            for (int64_t i = 0; i < source.num_elements; ++i)
                EXPECT_EQ(broadcast_float_bits(got[i]), broadcast_float_bits(expected[i]))
                    << "element " << i;
        }

        TEST_F(Unit_SpatialBroadcastTest, LayoutConvertMovesTheChannelIndex)
        {
            // A round trip alone cannot catch a conversion that permuted
            // nothing, so this one reads the intermediate buffer directly.
            const int64_t N = 1, H = 2, W = 2, C = 2;

            auto source = ramp({N, H, W, C});
            auto nchw_storage = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto nchw = nchw_view(nchw_storage, N, H, W, C);

            engine->spatial().layout_convert(source, nchw);
            engine->execute();

            // Input element (h, w, c) sits at h*4 + w*2 + c and holds that
            // number; in NCHW it must land at c*4 + h*2 + w.
            const std::vector<float> expected = {0, 2, 4, 6,    // channel 0
                                                 1, 3, 5, 7};   // channel 1

            const float* got = nchw_storage.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "NCHW element " << i;
        }

        TEST_F(Unit_SpatialBroadcastTest, TheSyclKernelsAgreeWithTheReferenceKernels)
        {
            // Compared here rather than under ExecutionMode::VERIFY, which runs
            // both kernels but lets the reference overwrite the device's answer
            // instead of checking it (see Internal::dispatch_with_mode).
            const int64_t N = 2, H = 5, W = 3, C = 4;

            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                p[i] = static_cast<float>((i * 37) % 23) - 11.0f;

            auto gamma = engine->create_tensor({C}, Core::DataType::FLOAT32);
            auto beta = engine->create_tensor({C}, Core::DataType::FLOAT32);
            float* g = gamma.data_as<float>();
            float* b = beta.data_as<float>();
            for (int64_t c = 0; c < C; ++c)
            {
                g[c] = 0.5f + static_cast<float>(c) * 0.25f;
                b[c] = static_cast<float>(c) - 1.5f;
            }

            auto plane = engine->create_tensor({N, C}, Core::DataType::FLOAT32);
            float* pl = plane.data_as<float>();
            for (int64_t i = 0; i < plane.num_elements; ++i)
                pl[i] = static_cast<float>((i * 13) % 7) - 3.0f;

            auto affine_reference = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto affine_device = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto spread_reference = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto spread_device = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto convert_reference_storage =
                engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto convert_device_storage =
                engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            auto convert_reference = nchw_view(convert_reference_storage, N, H, W, C);
            auto convert_device = nchw_view(convert_device_storage, N, H, W, C);

            engine->set_mode(ExecutionMode::FORCE_REFERENCE);
            engine->spatial().channel_affine(in, gamma, beta, affine_reference);
            engine->spatial().broadcast_spatial(plane, spread_reference, 1.0 / 15.0);
            engine->spatial().layout_convert(in, convert_reference);
            engine->execute();

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            engine->spatial().channel_affine(in, gamma, beta, affine_device);
            engine->spatial().broadcast_spatial(plane, spread_device, 1.0 / 15.0);
            engine->spatial().layout_convert(in, convert_device);
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
