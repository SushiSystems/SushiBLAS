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
 * @file test_spatial_broadcast.cpp
 * @brief Tests SpatialOps::channel_affine, broadcast_spatial and layout_convert.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns count values 0, 1, 2 and so on. */
            std::vector<float> ramp_values(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
                return values;
            }

            /** @brief Returns a float tensor of the given shape holding values in memory order. */
            Tensor filled(Engine& engine, std::initializer_list<int64_t> dims,
                          const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor(dims, Core::DataType::Float32);
                fill_tensor(t, values);
                return t;
            }

            /** @brief Returns a float tensor of the given shape whose element k holds `k`. */
            Tensor ramp(Engine& engine, std::initializer_list<int64_t> dims)
            {
                Tensor t = engine.create_tensor(dims, Core::DataType::Float32);
                fill_tensor(t, ramp_values(t.num_elements));
                return t;
            }

            /** @brief Returns a float tensor of the given shape whose every element is value. */
            Tensor uniform(Engine& engine, std::initializer_list<int64_t> dims, float value)
            {
                Tensor t = engine.create_tensor(dims, Core::DataType::Float32);
                fill_tensor(t, std::vector<float>(static_cast<std::size_t>(t.num_elements), value));
                return t;
            }

            /**
             * @brief Returns an NCHW-strided view of base, logically [N, H, W, C].
             * @param base Contiguous, at least N*H*W*C elements; the view borrows its storage.
             * @return The view, whose channel stride is h*w and whose column stride is 1.
             * @see tests/operations/spatial/README.md
             */
            Tensor nchw_view(const Tensor& base, int64_t n, int64_t h, int64_t w, int64_t c)
            {
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> s{};
                s[0] = c * h * w;
                s[1] = w;
                s[2] = 1;
                s[3] = h * w;

                const int64_t dims[4] = {n, h, w, c};
                return Tensor(base.storage, SushiRuntime::span<const int64_t>(dims, dims + 4), s,
                              Core::DataType::Float32, base.storage_offset, base.layout);
            }

            /** @brief Returns the NCHW memory order of a ramp laid out NHWC, element by element. */
            std::vector<float> nchw_order_of_nhwc_ramp(int64_t N, int64_t H, int64_t W, int64_t C)
            {
                std::vector<float> nchw(static_cast<std::size_t>(N * H * W * C));
                for (int64_t n = 0; n < N; ++n)
                    for (int64_t h = 0; h < H; ++h)
                        for (int64_t w = 0; w < W; ++w)
                            for (int64_t c = 0; c < C; ++c)
                                nchw[static_cast<std::size_t>(((n * C + c) * H + h) * W + w)] =
                                    static_cast<float>(((n * H + h) * W + w) * C + c);
                return nchw;
            }

            /** @brief Returns each (n, c) value of plane copied over H x W pixels and scaled. */
            std::vector<float> spread_plane(const std::vector<float>& plane, int64_t N, int64_t H,
                                            int64_t W, int64_t C, float scale)
            {
                std::vector<float> spread(static_cast<std::size_t>(N * H * W * C));
                for (std::size_t i = 0; i < spread.size(); ++i)
                {
                    const int64_t c = static_cast<int64_t>(i) % C;
                    const int64_t n = static_cast<int64_t>(i) / (H * W * C);
                    spread[i] = plane[static_cast<std::size_t>(n * C + c)] * scale;
                }
                return spread;
            }

            /** @brief Returns the sum of every (n, c) plane of an NHWC tensor, in float. */
            std::vector<float> plane_sums(const std::vector<float>& nhwc, int64_t N, int64_t H,
                                          int64_t W, int64_t C)
            {
                std::vector<float> sums(static_cast<std::size_t>(N * C), 0.0f);
                for (int64_t n = 0; n < N; ++n)
                    for (int64_t h = 0; h < H; ++h)
                        for (int64_t w = 0; w < W; ++w)
                            for (int64_t c = 0; c < C; ++c)
                                sums[static_cast<std::size_t>(n * C + c)] +=
                                    nhwc[static_cast<std::size_t>(((n * H + h) * W + w) * C + c)];
                return sums;
            }

            /** @brief Returns count values of a reproducible pattern, offset down by shift. */
            std::vector<float> pattern_values(int64_t count, int64_t factor, int64_t modulus,
                                              float shift)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] =
                        static_cast<float>((i * factor) % modulus) - shift;
                return values;
            }

            /** @brief Names the extents of one layout_convert round trip. */
            struct ConvertExtents
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The images. */
                int64_t n;

                /** @brief The rows. */
                int64_t h;

                /** @brief The columns. */
                int64_t w;

                /** @brief The channels. */
                int64_t c;
            };

            /** @brief Names a row by its name. */
            std::string convert_row_name(const ::testing::TestParamInfo<ConvertExtents>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the broadcast and layout operations on the host engine. */
        class Small_SpatialBroadcastTest : public HostEngineTest
        {
        };

        /** @brief Round-trips layout_convert over one set of extents on the host engine. */
        class Small_SpatialLayoutRoundTripTest : public HostEngineTest,
                                                 public ::testing::WithParamInterface<ConvertExtents>
        {
        };

        /** @brief Runs the broadcast and layout operations on the device and the reference path. */
        class Medium_SpatialBroadcastTest : public SushiBLASTest
        {
        };

        TEST_F(Small_SpatialBroadcastTest, ChannelAffineAppliesThePerChannelPair)
        {
            Tensor in = ramp(*engine, {1, 2, 2, 3});
            Tensor gamma = filled(*engine, {3}, {2.0f, 3.0f, 4.0f});
            Tensor beta = filled(*engine, {3}, {10.0f, 20.0f, 30.0f});
            Tensor out = engine->create_tensor({1, 2, 2, 3}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).channel_affine(in, gamma, beta, out);
            ASSERT_TRUE(recording->run_succeeds());

            // Element k has channel k % 3, so it becomes k * gamma[k % 3] + beta[k % 3].
            expect_tensor_equal(out, {10, 23, 38,
                                      16, 32, 50,
                                      22, 41, 62,
                                      28, 50, 74});
        }

        TEST_F(Small_SpatialBroadcastTest, ChannelAffineWithUnitGammaAndZeroBetaIsACopy)
        {
            Tensor in = ramp(*engine, {2, 3, 4, 5});
            Tensor gamma = uniform(*engine, {5}, 1.0f);
            Tensor beta = uniform(*engine, {5}, 0.0f);
            Tensor out = engine->create_tensor({2, 3, 4, 5}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).channel_affine(in, gamma, beta, out);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, ramp_values(in.num_elements));
        }

        TEST_F(Small_SpatialBroadcastTest, BroadcastSpatialSpreadsAndScales)
        {
            const int64_t N = 2, H = 4, W = 5, C = 3;
            const std::vector<float> plane = {1, 2, 3,
                                              4, 5, 6};
            Tensor in = filled(*engine, {N, C}, plane);
            Tensor out = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).broadcast_spatial(in, out, 0.5);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, spread_plane(plane, N, H, W, C, 0.5f));
        }

        TEST_F(Small_SpatialBroadcastTest, BroadcastSpatialAtOneOverHwGivesEveryPixelTheSameShare)
        {
            const int64_t N = 2, H = 4, W = 5, C = 3;
            const double scale = 1.0 / static_cast<double>(H * W);
            Tensor in = uniform(*engine, {N, C}, 1.0f);
            Tensor out = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).broadcast_spatial(in, out, scale);
            ASSERT_TRUE(recording->run_succeeds());

            // float(1/20) is one rounding of an exact double, so every element is identical.
            expect_tensor_equal(out, std::vector<float>(static_cast<std::size_t>(out.num_elements),
                                                        static_cast<float>(scale)));
        }

        TEST_F(Small_SpatialBroadcastTest, BroadcastSpatialAtOneOverHwSumsEachPlaneBackToOne)
        {
            const int64_t N = 2, H = 4, W = 5, C = 3;
            Tensor in = uniform(*engine, {N, C}, 1.0f);
            Tensor out = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).broadcast_spatial(
                in, out, 1.0 / static_cast<double>(H * W));
            ASSERT_TRUE(recording->run_succeeds());

            // Each plane adds H * W float terms, one rounding each. See the test README.
            expect_tensor(filled(*engine, {N, C}, plane_sums(read_tensor<float>(out), N, H, W, C)),
                          std::vector<float>(static_cast<std::size_t>(N * C), 1.0f),
                          absolute_tolerance<float>(static_cast<std::size_t>(H * W)));
        }

        TEST_F(Small_SpatialBroadcastTest, LayoutConvertMovesTheChannelIndex)
        {
            // A round trip cannot catch a no-op conversion, so the intermediate buffer is read.
            const int64_t N = 1, H = 2, W = 2, C = 2;
            Tensor source = ramp(*engine, {N, H, W, C});
            Tensor nchw_storage = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor nchw = nchw_view(nchw_storage, N, H, W, C);

            SushiBLAS::SpatialOps(recorder()).layout_convert(source, nchw);
            ASSERT_TRUE(recording->run_succeeds());

            // Element (h, w, c) sits at h*4 + w*2 + c and must land at c*4 + h*2 + w in NCHW.
            expect_tensor_equal(nchw_storage, {0, 2, 4, 6,
                                               1, 3, 5, 7});
        }

        TEST_F(Small_SpatialBroadcastTest, LayoutConvertMovesRaggedTilesToNCHW)
        {
            // No extent is a multiple of the 32-wide tile, so every tile edge is ragged.
            const int64_t N = 2, H = 5, W = 7, C = 37;
            Tensor source = ramp(*engine, {N, H, W, C});
            Tensor nchw_storage = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor nchw = nchw_view(nchw_storage, N, H, W, C);

            SushiBLAS::SpatialOps(recorder()).layout_convert(source, nchw);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(nchw_storage, nchw_order_of_nhwc_ramp(N, H, W, C));
        }

        TEST_P(Small_SpatialLayoutRoundTripTest, LayoutConvertRoundTripsToItself)
        {
            // NHWC to NCHW and back: the same call, with the permuted strides on the other operand.
            const ConvertExtents& e = GetParam();
            Tensor source = ramp(*engine, {e.n, e.h, e.w, e.c});
            Tensor middle_storage =
                engine->create_tensor({e.n, e.h, e.w, e.c}, Core::DataType::Float32);
            Tensor middle = nchw_view(middle_storage, e.n, e.h, e.w, e.c);
            Tensor destination =
                engine->create_tensor({e.n, e.h, e.w, e.c}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).layout_convert(source, middle);
            SushiBLAS::SpatialOps(recorder()).layout_convert(middle, destination);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(destination, ramp_values(source.num_elements));
        }

        INSTANTIATE_TEST_SUITE_P(Extents, Small_SpatialLayoutRoundTripTest,
                                 ::testing::Values(ConvertExtents{"Small", 2, 3, 4, 5},
                                                   ConvertExtents{"RaggedTiles", 2, 5, 7, 37}),
                                 convert_row_name);

        TEST_F(Medium_SpatialBroadcastTest, TheSYCLChannelAffineAgreesWithTheReferenceChannelAffine)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            const int64_t N = 2, H = 5, W = 3, C = 4;
            Tensor in = filled(*engine, {N, H, W, C}, pattern_values(N * H * W * C, 37, 23, 11.0f));
            Tensor gamma = filled(*engine, {C}, {0.5f, 0.75f, 1.0f, 1.25f});
            Tensor beta = filled(*engine, {C}, {-1.5f, -0.5f, 0.5f, 1.5f});
            Tensor from_reference = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor from_device = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).channel_affine(in, gamma, beta, from_reference);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).channel_affine(in, gamma, beta, from_device);
            recording->run().value();

            expect_tensor_equal(from_device, read_tensor<float>(from_reference));
        }

        TEST_F(Medium_SpatialBroadcastTest, TheSYCLBroadcastSpatialAgreesWithTheReferenceBroadcastSpatial)
        {
            const int64_t N = 2, H = 5, W = 3, C = 4;
            Tensor plane = filled(*engine, {N, C}, pattern_values(N * C, 13, 7, 3.0f));
            Tensor from_reference = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor from_device = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).broadcast_spatial(plane, from_reference, 1.0 / 15.0);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).broadcast_spatial(plane, from_device, 1.0 / 15.0);
            recording->run().value();

            expect_tensor_equal(from_device, read_tensor<float>(from_reference));
        }

        TEST_F(Medium_SpatialBroadcastTest, TheSYCLLayoutConvertAgreesWithTheReferenceLayoutConvert)
        {
            const int64_t N = 2, H = 5, W = 3, C = 4;
            Tensor in = filled(*engine, {N, H, W, C}, pattern_values(N * H * W * C, 37, 23, 11.0f));
            Tensor reference_storage = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor device_storage = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            Tensor reference_view = nchw_view(reference_storage, N, H, W, C);
            Tensor device_view = nchw_view(device_storage, N, H, W, C);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).layout_convert(in, reference_view);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).layout_convert(in, device_view);
            recording->run().value();

            expect_tensor_equal(device_storage, read_tensor<float>(reference_storage));
        }
    } // namespace Test
} // namespace SushiBLAS
