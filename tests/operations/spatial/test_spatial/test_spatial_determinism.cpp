/**************************************************************************/
/* test_spatial_determinism.cpp                                           */
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
 * @file test_spatial_determinism.cpp
 * @brief Tests that the spatial backward kernels give identical bits on two runs.
 * @author Mustafa Garip
 * @see tests/operations/spatial/README.md
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The bit pattern of a float; `std::bit_cast` is C++20. */
            uint32_t determinism_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }

            /**
             * @brief Fills a buffer with values whose sum is order-sensitive.
             * @see tests/operations/spatial/README.md
             */
            void fill_order_sensitive(float* p, int64_t count)
            {
                for (int64_t i = 0; i < count; ++i)
                {
                    switch (i % 4)
                    {
                        case 0: p[i] = 1e8f; break;
                        case 1: p[i] = 1.0f; break;
                        case 2: p[i] = -1e8f; break;
                        default: p[i] = 1.0f; break;
                    }
                }
            }

            /** @brief Fills a buffer with ones, so a product with it keeps the other factor. */
            void fill_ones(float* p, int64_t count)
            {
                for (int64_t i = 0; i < count; ++i)
                    p[i] = 1.0f;
            }

            /**
             * @brief Runs @p run twice and asserts the two results agree bit for bit.
             * @tparam Run A callable `(std::vector<uint32_t>&) -> void` that writes one pass's bits.
             * @param elements How many elements the result holds.
             */
            template<typename Run>
            void expect_bit_identical_across_runs(Run&& run, int64_t elements)
            {
                std::vector<uint32_t> first(static_cast<std::size_t>(elements));
                std::vector<uint32_t> second(static_cast<std::size_t>(elements));

                run(first);
                run(second);

                for (std::size_t i = 0; i < first.size(); ++i)
                    ASSERT_EQ(first[i], second[i])
                        << "element " << i << " differs between two runs of the same kernel";
            }
        } // namespace

        class Medium_SpatialDeterminismTest : public SushiBLASTest
        {
            protected:
                /** @brief Copies a float tensor's bit patterns into @p out. */
                static void capture(const Tensor& t, std::vector<uint32_t>& out)
                {
                    const float* p = t.data_as<float>();
                    for (std::size_t i = 0; i < out.size(); ++i)
                        out[i] = determinism_float_bits(p[i]);
                }
        };

        TEST_F(Medium_SpatialDeterminismTest, Col2ImIsBitReproducible)
        {
            // 3x3, stride 1, padded by one: every interior pixel gathers nine terms.
            const int64_t N = 2, H = 6, W = 7, C = 3;

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto cols = engine->create_tensor({N * out_h * out_w, 3 * 3 * C},
                                                      Core::DataType::Float32);
                    fill_order_sensitive(cols.data_as<float>(), cols.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
                    SushiBLAS::SpatialOps(recorder()).col2im(cols, dx, g, out_h, out_w);
                    recording->run().value();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Medium_SpatialDeterminismTest, MaxPoolBackwardIsBitReproducible)
        {
            // 2x2, stride 1: a pixel selected by several windows accumulates all their gradients.
            const int64_t N = 2, H = 6, W = 7, C = 3;

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 1);
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 1);

            // One dominant pixel per channel plane, so every window in reach selects it.
            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                p[i] = 0.0f;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t c = 0; c < C; ++c)
                    p[((n * H + 2) * W + 3) * C + c] = 1.0f;

            auto indices = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);
            auto pooled = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(recorder()).pool_max(in, pooled, indices, g);
            recording->run().value();

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto dy = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
                    SushiBLAS::SpatialOps(recorder()).pool_max_backward(dy, indices, dx, g);
                    recording->run().value();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Medium_SpatialDeterminismTest, AvgPoolBackwardIsBitReproducible)
        {
            // 3x3 stride 1: every interior pixel gathers nine shares.
            const int64_t N = 2, H = 6, W = 7, C = 3;

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;

            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 3, 1);

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto dy = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
                    SushiBLAS::SpatialOps(recorder()).pool_avg_backward(dy, dx, g);
                    recording->run().value();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Medium_SpatialDeterminismTest, WindowDotBackwardIsBitReproducible)
        {
            // 3x3, stride 1, padded by one, unit taps: every interior pixel gathers nine terms.
            const int64_t N = 2, H = 6, W = 7, C = 3;

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto dy = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto weights = engine->create_tensor({3, 3, C}, Core::DataType::Float32);
                    fill_ones(weights.data_as<float>(), weights.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
                    SushiBLAS::SpatialOps(recorder()).window_dot_backward(dy, weights, dx, g);
                    recording->run().value();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Medium_SpatialDeterminismTest, WindowDotWeightsBackwardIsBitReproducible)
        {
            // 3x3, stride 1, padded by one, unit input: every tap folds up to 84 gradients.
            const int64_t N = 2, H = 6, W = 7, C = 3;

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto dy = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto in = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
                    fill_ones(in.data_as<float>(), in.num_elements);

                    auto dweights = engine->create_tensor({3, 3, C}, Core::DataType::Float32);
                    SushiBLAS::SpatialOps(recorder())
                        .window_dot_weights_backward(dy, in, dweights, g);
                    recording->run().value();

                    capture(dweights, sink);
                },
                3 * 3 * C);
        }

        TEST_F(Medium_SpatialDeterminismTest, UpsampleBackwardIsBitReproducible)
        {
            // Scale 3 in both axes, so every input pixel gathers nine terms.
            const int64_t N = 2, H = 5, W = 4, C = 3;
            const int64_t scale = 3;

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto dy = engine->create_tensor({N, H * scale, W * scale, C},
                                                    Core::DataType::Float32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
                    SushiBLAS::SpatialOps(recorder())
                        .upsample_nearest_backward(dy, dx, scale, scale);
                    recording->run().value();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Medium_SpatialDeterminismTest, MeanSpatialIsBitReproducible)
        {
            // Few segments over a long extent, so the reduction folds through the halving tree.
            const int64_t N = 2, H = 256, W = 256, C = 2;

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto in = engine->create_tensor({N, H, W, C}, Core::DataType::Float32);
                    fill_order_sensitive(in.data_as<float>(), in.num_elements);

                    auto out = engine->create_tensor({N, C}, Core::DataType::Float32);
                    SushiBLAS::ReductionOps(recorder()).mean_spatial(in, out);
                    recording->run().value();

                    capture(out, sink);
                },
                N * C);
        }
    }
}
