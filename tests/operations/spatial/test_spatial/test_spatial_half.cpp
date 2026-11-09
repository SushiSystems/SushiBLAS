/**************************************************************************/
/* test_spatial_half.cpp                                                  */
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
 * @file test_spatial_half.cpp
 * @brief Tests that the spatial kernels sum HALF terms in FLOAT32 and agree across both modes.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the bit patterns of the HALF values. */
            std::vector<uint16_t> half_bit_patterns(const std::vector<sycl::half>& values)
            {
                static_assert(sizeof(sycl::half) == sizeof(uint16_t), "HALF is 16 bits wide");

                std::vector<uint16_t> bits(values.size());
                std::memcpy(bits.data(), values.data(), values.size() * sizeof(uint16_t));
                return bits;
            }

            /** @brief Returns count values of the pattern `((i * 37) % 23) - 11`, exact in HALF. */
            std::vector<float> half_pattern_values(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>((i * 37) % 23) - 11.0f;
                return values;
            }

            /** @brief Returns a HALF tensor of the given shape holding values in memory order. */
            Tensor half_tensor(Engine& engine, std::initializer_list<int64_t> dims,
                               const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor(dims, Core::DataType::Half);
                fill_tensor(t, values);
                return t;
            }

            /** @brief Returns a HALF tensor of the given shape with every element zero. */
            Tensor half_zeros(Engine& engine, SushiRuntime::span<const int64_t> dims)
            {
                Tensor t = engine.create_tensor(dims, Core::DataType::Half);
                fill_tensor(t, std::vector<float>(static_cast<std::size_t>(t.num_elements), 0.0f));
                return t;
            }

            /**
             * @brief Returns an NCHW-strided HALF view of base, logically [N, H, W, C].
             * @param base Contiguous, at least N*H*W*C elements; the view borrows its storage.
             */
            Tensor half_nchw_view(const Tensor& base, int64_t n, int64_t h, int64_t w, int64_t c)
            {
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> s{};
                s[0] = c * h * w;
                s[1] = w;
                s[2] = 1;
                s[3] = h * w;

                const int64_t dims[4] = {n, h, w, c};
                return Tensor(base.storage, SushiRuntime::span<const int64_t>(dims, dims + 4), s,
                              Core::DataType::Half, base.storage_offset, base.layout);
            }

            /** @brief Holds the inputs every routine of the agreement table reads. */
            struct HalfInputs
            {
                Tensor image;              /**< N x H x W x C pattern. */
                Tensor columns;            /**< The im2col matrix pattern. */
                Tensor pooled_gradient;    /**< The gradient of a pooled image. */
                Tensor upsampled_gradient; /**< The gradient of an upsampled image. */
                Tensor plane;              /**< N x C pattern. */
                Tensor gamma;              /**< The per-channel scales. */
                Tensor beta;               /**< The per-channel offsets. */
                Tensor indices;            /**< The argmax indices of pool_max over image. */
                Tensor scratch_values;     /**< The values a pool_max row writes and drops. */
                Tensor scratch_indices;    /**< The indices a pool_max row writes and drops. */
            };

            /** @brief The extents and geometry the agreement table runs on. */
            struct HalfCase
            {
                int64_t N = 2, H = 7, W = 5, C = 3;
                int64_t scale = 2, extra_channels = 2, channel_offset = 1;
                int64_t out_h = 0, out_w = 0, windows = 0, patch = 0;
                Spatial::ConvGeometry g;

                /** @brief Builds the geometry and the derived extents. */
                HalfCase()
                {
                    g.kernel_h = 3;
                    g.kernel_w = 2;
                    g.stride_h = 2;
                    g.stride_w = 1;
                    g.dilation_h = 2;
                    g.pad_top = 1;
                    g.pad_left = 1;
                    out_h = Spatial::output_extent(H, 1, 1, 2, 3, 2);
                    out_w = Spatial::output_extent(W, 1, 1, 1, 2, 1);
                    windows = N * out_h * out_w;
                    patch = g.kernel_h * g.kernel_w * C;
                }
            };

            /** @brief Names one routine, the shape of its result and how it records. */
            struct RoutineRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The shape of the result. */
                std::vector<int64_t> out_dims;

                /** @brief Records the routine into the result tensor. */
                std::function<void(SpatialOps&, HalfInputs&, Tensor&)> record;
            };

            /** @brief Returns the rows, one per spatial routine. */
            std::vector<RoutineRow> routine_rows()
            {
                const HalfCase k;
                return {
                    {"Im2Col", {k.windows, k.patch},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.im2col(in.image, out, k.g, k.out_h, k.out_w); }},
                    {"Col2Im", {k.N, k.H, k.W, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.col2im(in.columns, out, k.g, k.out_h, k.out_w); }},
                    {"PoolMax", {k.N, k.out_h, k.out_w, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.pool_max(in.image, out, in.scratch_indices, k.g); }},
                    {"PoolMaxBackward", {k.N, k.H, k.W, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.pool_max_backward(in.pooled_gradient, in.indices, out, k.g); }},
                    {"PoolAvg", {k.N, k.out_h, k.out_w, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.pool_avg(in.image, out, k.g); }},
                    {"PoolAvgBackward", {k.N, k.H, k.W, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.pool_avg_backward(in.pooled_gradient, out, k.g); }},
                    {"UpsampleNearest", {k.N, k.scale * k.H, k.scale * k.W, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.upsample_nearest(in.image, out, k.scale, k.scale); }},
                    {"UpsampleNearestBackward", {k.N, k.H, k.W, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.upsample_nearest_backward(in.upsampled_gradient, out, k.scale, k.scale); }},
                    {"ConcatChannels", {k.N, k.H, k.W, k.C + k.extra_channels},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.concat_channels(in.image, out, k.channel_offset); }},
                    {"SliceChannels", {k.N, k.H, k.W, k.C - k.channel_offset},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.slice_channels(in.image, k.channel_offset, out); }},
                    {"ChannelAffine", {k.N, k.H, k.W, k.C},
                     [](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.channel_affine(in.image, in.gamma, in.beta, out); }},
                    {"BroadcastSpatial", {k.N, k.H, k.W, k.C},
                     [](SpatialOps& s, HalfInputs& in, Tensor& out)
                     { s.broadcast_spatial(in.plane, out, 0.5); }},
                    {"LayoutConvert", {k.N, k.H, k.W, k.C},
                     [k](SpatialOps& s, HalfInputs& in, Tensor& out)
                     {
                         Tensor converted = half_nchw_view(out, k.N, k.H, k.W, k.C);
                         s.layout_convert(in.image, converted);
                     }}};
            }

            /** @brief Names a row by its name. */
            std::string routine_row_name(const ::testing::TestParamInfo<RoutineRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs spatial operations on HALF tensors on the host engine. */
        class Small_SpatialHalf : public HostEngineTest
        {
        };

        /** @brief Runs each spatial routine on HALF tensors under both forced modes. */
        class Medium_SpatialHalf : public SushiBLASTest,
                                   public ::testing::WithParamInterface<RoutineRow>
        {
            protected:
                /** @brief Builds the engine and the inputs, and skips on a device without fp16. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    const sycl::device device = TestEnvironment::queue().get_device();
                    if (!device.has(sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";

                    const HalfCase k;
                    inputs.image = half_tensor(*engine, {k.N, k.H, k.W, k.C},
                                               half_pattern_values(k.N * k.H * k.W * k.C));
                    inputs.columns = half_tensor(*engine, {k.windows, k.patch},
                                                 half_pattern_values(k.windows * k.patch));
                    inputs.pooled_gradient =
                        half_tensor(*engine, {k.N, k.out_h, k.out_w, k.C},
                                    half_pattern_values(k.N * k.out_h * k.out_w * k.C));
                    inputs.upsampled_gradient =
                        half_tensor(*engine, {k.N, k.scale * k.H, k.scale * k.W, k.C},
                                    half_pattern_values(k.N * k.scale * k.H * k.scale * k.W * k.C));
                    inputs.plane = half_tensor(*engine, {k.N, k.C}, half_pattern_values(k.N * k.C));
                    inputs.gamma = half_tensor(*engine, {k.C}, {1, 2, 3});
                    inputs.beta = half_tensor(*engine, {k.C}, {-1, 0, 1});
                    inputs.indices = engine->create_tensor({k.N, k.out_h, k.out_w, k.C},
                                                           Core::DataType::Int64);
                    inputs.scratch_indices = engine->create_tensor({k.N, k.out_h, k.out_w, k.C},
                                                                   Core::DataType::Int64);
                    inputs.scratch_values =
                        half_tensor(*engine, {k.N, k.out_h, k.out_w, k.C},
                                    std::vector<float>(
                                        static_cast<std::size_t>(k.N * k.out_h * k.out_w * k.C), 0.0f));

                    set_mode(ExecutionMode::ForceReference);
                    SushiBLAS::SpatialOps(recorder()).pool_max(inputs.image, inputs.scratch_values,
                                                               inputs.indices, k.g);
                    recording->run().value();
                }

                /** @brief The inputs of the routine under test. */
                HalfInputs inputs;
        };

        /** @brief Tests that col2im adds 2048 + 1 + 1 to 2050, where a HALF sum stalls at 2048. */
        TEST_F(Small_SpatialHalf, Col2ImSumsItsTermsInFloat)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 3;
            g.pad_left = 1;

            // Pixel 2 gathers (window 3, tap 0), (window 2, tap 1), (window 1, tap 2).
            Tensor cols = half_tensor(*engine, {5, 3}, {0, 0, 0,
                                                        0, 0, 1,
                                                        0, 1, 0,
                                                        2048, 0, 0,
                                                        0, 0, 0});
            Tensor image = half_tensor(*engine, {1, 1, 5, 1}, {0, 0, 0, 0, 0});

            SushiBLAS::SpatialOps(recorder()).col2im(cols, image, g, 1, 5);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(image, {0, 0, 2050, 0, 0});
        }

        /** @brief Tests that pool_max_backward adds the gradients of one pixel in FLOAT32. */
        TEST_F(Small_SpatialHalf, PoolMaxBackwardSumsItsTermsInFloat)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 3;
            g.pad_left = 1;

            // Windows 1, 2 and 3 select pixel 2, which gathers them as 3, 2, 1.
            Tensor in = half_tensor(*engine, {1, 1, 5, 1}, {0, 0, 9, 0, 0});
            Tensor out = half_tensor(*engine, {1, 1, 5, 1}, {0, 0, 0, 0, 0});
            Tensor indices = engine->create_tensor({1, 1, 5, 1}, Core::DataType::Int64);
            Tensor dy = half_tensor(*engine, {1, 1, 5, 1}, {0, 1, 1, 2048, 0});
            Tensor dx = half_tensor(*engine, {1, 1, 5, 1}, {0, 0, 0, 0, 0});

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, g);
            SushiBLAS::SpatialOps(recorder()).pool_max_backward(dy, indices, dx, g);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, {0, 0, 2050, 0, 0});
        }

        /** @brief Tests that upsample_nearest_backward adds a 2 by 2 block in FLOAT32. */
        TEST_F(Small_SpatialHalf, UpsampleBackwardSumsItsBlockInFloat)
        {
            Tensor dy = half_tensor(*engine, {1, 2, 2, 1}, {2048, 1,
                                                            1, 0});
            Tensor dx = half_tensor(*engine, {1, 1, 1, 1}, {0});

            SushiBLAS::SpatialOps(recorder()).upsample_nearest_backward(dy, dx, 2, 2);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, {2050});
        }

        /** @brief Tests that pool_avg divides the FLOAT32 sum 2050 by 4 and returns 512.5. */
        TEST_F(Small_SpatialHalf, PoolAvgDividesAFloatSum)
        {
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            Tensor in = half_tensor(*engine, {1, 2, 2, 1}, {2048, 1,
                                                            1, 0});
            Tensor out = half_tensor(*engine, {1, 1, 1, 1}, {0});

            SushiBLAS::SpatialOps(recorder()).pool_avg(in, out, g);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, {512.5f});
        }

        /** @brief Tests that pool_avg_backward adds its shares in FLOAT32 and rounds once. */
        TEST_F(Small_SpatialHalf, PoolAvgBackwardAddsItsSharesInFloat)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 4;
            g.pad_left = 3;

            // Pixel w gathers the shares dy / 4 of windows w + 3 down to w.
            Tensor dy = half_tensor(*engine, {1, 1, 7, 1}, {0, 4, 4, 8192, 0, 0, 0});
            Tensor dx = half_tensor(*engine, {1, 1, 4, 1}, {0, 0, 0, 0});

            SushiBLAS::SpatialOps(recorder()).pool_avg_backward(dy, dx, g);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, {2050, 2050, 2048, 2048});
        }

        /** @brief Tests that pool_max of two HALF negatives is the larger element. */
        TEST_F(Small_SpatialHalf, PoolMaxOfTwoHalfNegativesIsTheLargerElement)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 2;
            Tensor in = half_tensor(*engine, {1, 1, 2, 1}, {-60000, -65504});
            Tensor out = half_tensor(*engine, {1, 1, 1, 1}, {0});
            Tensor indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, g);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, {-60000});
        }

        /** @brief Tests that a spatial kernel writes the same HALF bits as its reference. */
        TEST_P(Medium_SpatialHalf, TheKernelAgreesWithItsReferenceInHalf)
        {
            const RoutineRow& row = GetParam();
            const SushiRuntime::span<const int64_t> dims(row.out_dims.data(), row.out_dims.size());
            Tensor from_reference = half_zeros(*engine, dims);
            Tensor from_device = half_zeros(*engine, dims);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps reference_ops(recorder());
            row.record(reference_ops, inputs, from_reference);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps device_ops(recorder());
            row.record(device_ops, inputs, from_device);
            recording->run().value();

            EXPECT_EQ(half_bit_patterns(read_tensor<sycl::half>(from_device)),
                      half_bit_patterns(read_tensor<sycl::half>(from_reference)));
        }

        INSTANTIATE_TEST_SUITE_P(Routines, Medium_SpatialHalf, ::testing::ValuesIn(routine_rows()),
                                 routine_row_name);
    } // namespace Test
} // namespace SushiBLAS
