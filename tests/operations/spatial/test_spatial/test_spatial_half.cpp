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
 * @file test_spatial_half.cpp
 * @brief Tests that the spatial kernels sum HALF terms in FLOAT32 and agree across both modes.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSYCL,
                                                        ExecutionMode::ForceReference};

            /** @brief Returns the name a failure message gives a forced mode. */
            const char* forced_mode_name(ExecutionMode mode) noexcept
            {
                return mode == ExecutionMode::ForceReference ? "ForceReference" : "ForceSYCL";
            }

            /** @brief Returns the bit pattern of a HALF value. */
            uint16_t half_bits(sycl::half value) noexcept
            {
                static_assert(sizeof(sycl::half) == sizeof(uint16_t), "HALF is 16 bits wide");

                uint16_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }

            /**
             * @brief Holds the two results one operation writes, one per forced mode.
             */
            struct ModeOutputs
            {
                /** @brief Returns the tensor the given forced mode writes. */
                [[nodiscard]] Tensor& in(ExecutionMode mode) noexcept
                {
                    return mode == ExecutionMode::ForceReference ? reference : device;
                }

                const char* operation = ""; /**< The operation's name in a failure message. */
                Tensor reference;           /**< The result under ForceReference. */
                Tensor device;              /**< The result under ForceSYCL. */
            };
        } // namespace

        /**
         * @brief Runs spatial operations on HALF tensors in a chosen execution mode.
         */
        class Medium_SpatialHalf : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without fp16. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    const sycl::device device =
                        TestEnvironment::queue().get_device();
                    if (!device.has(sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }

                /**
                 * @brief Returns a HALF tensor whose element i holds `value_at(i)`.
                 * @param value_at Callable `(int64_t) -> float`; each value must be exact in HALF.
                 */
                template<typename ValueAt>
                [[nodiscard]] Tensor half_filled(std::initializer_list<int64_t> dims,
                                                 ValueAt value_at)
                {
                    Tensor t = engine->create_tensor(dims, Core::DataType::Half);

                    std::vector<sycl::half> host(static_cast<std::size_t>(t.num_elements));
                    for (int64_t i = 0; i < t.num_elements; ++i)
                        host[static_cast<std::size_t>(i)] = sycl::half(value_at(i));

                    SushiBLAS::IO().from_vector(t, host);
                    return t;
                }

                /** @brief Returns a HALF tensor holding values in memory order. */
                [[nodiscard]] Tensor half_tensor(std::initializer_list<int64_t> dims,
                                                 const std::vector<float>& values)
                {
                    return half_filled(dims, [&values](int64_t i)
                    {
                        return values[static_cast<std::size_t>(i)];
                    });
                }

                /** @brief Returns a HALF tensor with every element zero. */
                [[nodiscard]] Tensor half_zeros(std::initializer_list<int64_t> dims)
                {
                    return half_filled(dims, [](int64_t) { return 0.0f; });
                }

                /** @brief Returns a HALF tensor whose element i holds `((i * 37) % 23) - 11`. */
                [[nodiscard]] Tensor half_pattern(std::initializer_list<int64_t> dims)
                {
                    return half_filled(dims, [](int64_t i)
                    {
                        return static_cast<float>((i * 37) % 23) - 11.0f;
                    });
                }

                /** @brief Returns two zeroed HALF tensors of dims, one for each forced mode. */
                [[nodiscard]] ModeOutputs half_outputs(const char* operation,
                                                       std::initializer_list<int64_t> dims)
                {
                    ModeOutputs outputs;
                    outputs.operation = operation;
                    outputs.reference = half_zeros(dims);
                    outputs.device = half_zeros(dims);
                    return outputs;
                }

                /**
                 * @brief Returns an NCHW-strided HALF view of base, logically [N, H, W, C].
                 * @param base Contiguous, at least N*H*W*C elements; the view borrows its storage.
                 */
                [[nodiscard]] Tensor nchw_view(const Tensor& base, int64_t n, int64_t h, int64_t w,
                                               int64_t c)
                {
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> s{};
                    s[0] = c * h * w;
                    s[1] = w;
                    s[2] = 1;
                    s[3] = h * w;

                    const int64_t dims[4] = {n, h, w, c};
                    return Tensor(base.storage,
                                  SushiRuntime::span<const int64_t>(dims, dims + 4), s,
                                  Core::DataType::Half, base.storage_offset, base.layout);
                }

                /**
                 * @brief Expects a HALF tensor to hold exactly the expected values, bit for bit.
                 * @param expected Every value must be exact in HALF.
                 */
                void expect_values(const Tensor& t, const std::vector<float>& expected)
                {
                    const std::vector<sycl::half> got =
                        SushiBLAS::IO().to_vector<sycl::half>(t);

                    ASSERT_EQ(got.size(), expected.size());
                    for (std::size_t i = 0; i < expected.size(); ++i)
                        EXPECT_EQ(half_bits(got[i]), half_bits(sycl::half(expected[i])))
                            << "element " << i << ": expected " << expected[i]
                            << ", got " << static_cast<float>(got[i]);
                }

                /** @brief Expects the two modes to have written the same HALF bits. */
                void expect_equal_bits(const ModeOutputs& outputs)
                {
                    const std::vector<sycl::half> expected =
                        SushiBLAS::IO().to_vector<sycl::half>(outputs.reference);
                    const std::vector<sycl::half> got =
                        SushiBLAS::IO().to_vector<sycl::half>(outputs.device);

                    ASSERT_EQ(got.size(), expected.size()) << outputs.operation;
                    for (std::size_t i = 0; i < expected.size(); ++i)
                        EXPECT_EQ(half_bits(got[i]), half_bits(expected[i]))
                            << outputs.operation << " element " << i << ": reference "
                            << static_cast<float>(expected[i]) << ", device "
                            << static_cast<float>(got[i]);
                }
        };

        /** @brief Tests that col2im adds 2048 + 1 + 1 to 2050, where a HALF sum stalls at 2048. */
        TEST_F(Medium_SpatialHalf, Col2ImSumsItsTermsInFloat)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 3;
            g.pad_left = 1;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                // Pixel 2 gathers (window 3, tap 0), (window 2, tap 1), (window 1, tap 2).
                Tensor cols = half_tensor({5, 3}, {0, 0, 0,
                                                   0, 0, 1,
                                                   0, 1, 0,
                                                   2048, 0, 0,
                                                   0, 0, 0});
                Tensor image = half_zeros({1, 1, 5, 1});

                SushiBLAS::SpatialOps(recorder()).col2im(cols, image, g, 1, 5);
                recording->run().value();

                expect_values(image, {0, 0, 2050, 0, 0});
            }
        }

        /** @brief Tests that pool_max_backward adds the gradients of one pixel in FLOAT32. */
        TEST_F(Medium_SpatialHalf, PoolMaxBackwardSumsItsTermsInFloat)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 3;
            g.pad_left = 1;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                // Windows 1, 2 and 3 select pixel 2, which gathers them as 3, 2, 1.
                Tensor in = half_tensor({1, 1, 5, 1}, {0, 0, 9, 0, 0});
                Tensor out = half_zeros({1, 1, 5, 1});
                Tensor indices = engine->create_tensor({1, 1, 5, 1}, Core::DataType::Int64);
                Tensor dy = half_tensor({1, 1, 5, 1}, {0, 1, 1, 2048, 0});
                Tensor dx = half_zeros({1, 1, 5, 1});

                SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, g);
                SushiBLAS::SpatialOps(recorder()).pool_max_backward(dy, indices, dx, g);
                recording->run().value();

                expect_values(dx, {0, 0, 2050, 0, 0});
            }
        }

        /** @brief Tests that upsample_nearest_backward adds a 2 by 2 block in FLOAT32. */
        TEST_F(Medium_SpatialHalf, UpsampleBackwardSumsItsBlockInFloat)
        {
            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                Tensor dy = half_tensor({1, 2, 2, 1}, {2048, 1,
                                                       1, 0});
                Tensor dx = half_zeros({1, 1, 1, 1});

                SushiBLAS::SpatialOps(recorder()).upsample_nearest_backward(dy, dx, 2, 2);
                recording->run().value();

                expect_values(dx, {2050});
            }
        }

        /** @brief Tests that pool_avg divides the FLOAT32 sum 2050 by 4 and returns 512.5. */
        TEST_F(Medium_SpatialHalf, PoolAvgDividesAFloatSum)
        {
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                Tensor in = half_tensor({1, 2, 2, 1}, {2048, 1,
                                                       1, 0});
                Tensor out = half_zeros({1, 1, 1, 1});

                SushiBLAS::SpatialOps(recorder()).pool_avg(in, out, g);
                recording->run().value();

                expect_values(out, {512.5f});
            }
        }

        /** @brief Tests that pool_avg_backward adds its shares in FLOAT32 and rounds once. */
        TEST_F(Medium_SpatialHalf, PoolAvgBackwardAddsItsSharesInFloat)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 4;
            g.pad_left = 3;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                // Pixel w gathers the shares dy / 4 of windows w + 3 down to w.
                Tensor dy = half_tensor({1, 1, 7, 1}, {0, 4, 4, 8192, 0, 0, 0});
                Tensor dx = half_zeros({1, 1, 4, 1});

                SushiBLAS::SpatialOps(recorder()).pool_avg_backward(dy, dx, g);
                recording->run().value();

                expect_values(dx, {2050, 2050, 2048, 2048});
            }
        }

        /** @brief Tests that every spatial kernel writes the same HALF bits as its reference. */
        TEST_F(Medium_SpatialHalf, EveryKernelAgreesWithItsReferenceInHalf)
        {
            const int64_t N = 2, H = 7, W = 5, C = 3;
            const int64_t scale = 2, extra_channels = 2, channel_offset = 1;

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
            const int64_t windows = N * out_h * out_w;
            const int64_t patch = g.kernel_h * g.kernel_w * C;

            Tensor image = half_pattern({N, H, W, C});
            Tensor columns = half_pattern({windows, patch});
            Tensor pooled_gradient = half_pattern({N, out_h, out_w, C});
            Tensor upsampled_gradient = half_pattern({N, scale * H, scale * W, C});
            Tensor plane = half_pattern({N, C});
            Tensor gamma = half_tensor({C}, {1, 2, 3});
            Tensor beta = half_tensor({C}, {-1, 0, 1});

            ModeOutputs im2col = half_outputs("im2col", {windows, patch});
            ModeOutputs col2im = half_outputs("col2im", {N, H, W, C});
            ModeOutputs pool_max = half_outputs("pool_max", {N, out_h, out_w, C});
            ModeOutputs pool_max_backward = half_outputs("pool_max_backward", {N, H, W, C});
            ModeOutputs pool_avg = half_outputs("pool_avg", {N, out_h, out_w, C});
            ModeOutputs pool_avg_backward = half_outputs("pool_avg_backward", {N, H, W, C});
            ModeOutputs upsample = half_outputs("upsample_nearest", {N, scale * H, scale * W, C});
            ModeOutputs upsample_backward = half_outputs("upsample_nearest_backward", {N, H, W, C});
            ModeOutputs concat = half_outputs("concat_channels", {N, H, W, C + extra_channels});
            ModeOutputs slice = half_outputs("slice_channels", {N, H, W, C - channel_offset});
            ModeOutputs affine = half_outputs("channel_affine", {N, H, W, C});
            ModeOutputs broadcast = half_outputs("broadcast_spatial", {N, H, W, C});
            ModeOutputs convert = half_outputs("layout_convert", {N, H, W, C});

            ModeOutputs indices;
            indices.reference = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);
            indices.device = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                set_mode(mode);
                SushiBLAS::SpatialOps spatial(recorder());
                Tensor converted = nchw_view(convert.in(mode), N, H, W, C);

                spatial.im2col(image, im2col.in(mode), g, out_h, out_w);
                spatial.col2im(columns, col2im.in(mode), g, out_h, out_w);
                spatial.pool_max(image, pool_max.in(mode), indices.in(mode), g);
                spatial.pool_max_backward(pooled_gradient, indices.in(mode),
                                          pool_max_backward.in(mode), g);
                spatial.pool_avg(image, pool_avg.in(mode), g);
                spatial.pool_avg_backward(pooled_gradient, pool_avg_backward.in(mode), g);
                spatial.upsample_nearest(image, upsample.in(mode), scale, scale);
                spatial.upsample_nearest_backward(upsampled_gradient, upsample_backward.in(mode),
                                                  scale, scale);
                spatial.concat_channels(image, concat.in(mode), channel_offset);
                spatial.slice_channels(image, channel_offset, slice.in(mode));
                spatial.channel_affine(image, gamma, beta, affine.in(mode));
                spatial.broadcast_spatial(plane, broadcast.in(mode), 0.5);
                spatial.layout_convert(image, converted);
                recording->run().value();
            }

            for (const ModeOutputs* outputs :
                 {&im2col, &col2im, &pool_max, &pool_max_backward, &pool_avg, &pool_avg_backward,
                  &upsample, &upsample_backward, &concat, &slice, &affine, &broadcast, &convert})
                expect_equal_bits(*outputs);

            EXPECT_EQ(SushiBLAS::IO().to_vector<int64_t>(indices.device),
                      SushiBLAS::IO().to_vector<int64_t>(indices.reference))
                << "pool_max indices";
        }

        /** @brief Tests that pool_max of a HALF window of large negatives is its larger element. */
        TEST_F(Medium_SpatialHalf, PoolMaxOfHalfStartsBelowEveryValue)
        {
            Spatial::ConvGeometry g;
            g.kernel_w = 2;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(forced_mode_name(mode));
                set_mode(mode);

                Tensor in = half_tensor({1, 1, 2, 1}, {-60000, -65504});
                Tensor out = half_zeros({1, 1, 1, 1});
                Tensor indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

                SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, g);
                recording->run().value();

                expect_values(out, {-60000});
            }
        }
    }
}
