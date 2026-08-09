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

// The spatial domain's bit-reproducibility gate.
//
// Every backward in the spatial domain is written as a gather so that each
// output element sums its own contributions in a fixed order (spec CV-6). The
// failure mode this guards is silent: `sycl::atomic_ref`'s floating-point add is
// correct to any tolerance a functional test would use and non-reproducible in
// the last bits, so a tolerance-based test passes while the library's
// bit-reproducibility contract - inherited from the PUBLIC link against
// `sushiruntime_deterministic_fp` - is quietly gone.
//
// Nothing in the unit tier catches that, which is why this lives in the
// regression tier and compares raw bit patterns. `EXPECT_FLOAT_EQ` would accept
// the very difference being looked for.
//
// Every case below uses **overlapping** windows. A non-overlapping geometry has
// one contribution per output element and would pass with an atomic add in
// place, proving nothing.

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

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
             *
             * A ramp of similar magnitudes can be reordered without changing a
             * single bit of the total, which would make every test in this file
             * pass vacuously. Alternating a huge magnitude with a small one
             * makes the sum genuinely non-associative in binary32: `1e8 + 1.0`
             * rounds straight back to `1e8`, so whether the `1.0` is added
             * before or after the `-1e8` decides whether it survives at all.
             *
             * @param p The buffer's first element.
             * @param count How many elements to fill.
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

            /**
             * @brief Runs @p run twice and asserts the two results agree bit for bit.
             *
             * @tparam Run A callable `(std::vector<uint32_t>&) -> void` that
             *         performs one whole recording-and-execution pass and writes
             *         the result's bit patterns into its argument.
             * @param run The pass to repeat.
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

        class Regression_SpatialDeterminismTest : public SushiBLASTest
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

        TEST_F(Regression_SpatialDeterminismTest, Col2ImIsBitReproducible)
        {
            // 3x3 stride 1 padded by one, so every interior pixel gathers nine
            // terms of alternating magnitude.
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
                                                      Core::DataType::FLOAT32);
                    fill_order_sensitive(cols.data_as<float>(), cols.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
                    engine->spatial().col2im(cols, dx, g, out_h, out_w);
                    engine->execute();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Regression_SpatialDeterminismTest, MaxPoolBackwardIsBitReproducible)
        {
            // 2x2 stride 1 - YOLOv3-tiny's pool. A pixel selected by several
            // overlapping windows accumulates all of their gradients, which is
            // the only place this operation adds anything at all.
            const int64_t N = 2, H = 6, W = 7, C = 3;

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 1);
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 1);

            // One dominant pixel per channel plane, so every window in reach
            // selects it and the gradients genuinely accumulate. A generic input
            // would give most pixels a single contributor.
            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i) p[i] = 0.0f;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t c = 0; c < C; ++c)
                    p[((n * H + 2) * W + 3) * C + c] = 1.0f;

            auto indices = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::INT64);
            auto pooled = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::FLOAT32);
            engine->spatial().pool_max(in, pooled, indices, g);
            engine->execute();

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto dy = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::FLOAT32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
                    engine->spatial().pool_max_backward(dy, indices, dx, g);
                    engine->execute();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Regression_SpatialDeterminismTest, AvgPoolBackwardIsBitReproducible)
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
                    auto dy = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::FLOAT32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
                    engine->spatial().pool_avg_backward(dy, dx, g);
                    engine->execute();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Regression_SpatialDeterminismTest, UpsampleBackwardIsBitReproducible)
        {
            // Scale 3 in both axes, so every input pixel gathers nine terms.
            const int64_t N = 2, H = 5, W = 4, C = 3;
            const int64_t scale = 3;

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto dy = engine->create_tensor({N, H * scale, W * scale, C},
                                                    Core::DataType::FLOAT32);
                    fill_order_sensitive(dy.data_as<float>(), dy.num_elements);

                    auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
                    engine->spatial().upsample_nearest_backward(dy, dx, scale, scale);
                    engine->execute();

                    capture(dx, sink);
                },
                N * H * W * C);
        }

        TEST_F(Regression_SpatialDeterminismTest, MeanSpatialIsBitReproducible)
        {
            // Few segments over a long extent, so the reduction takes the
            // cooperative work-group-per-segment shape and folds through the
            // hand-written halving tree rather than one item's serial loop. That
            // is the shape whose combination order a `sycl::reduction` would
            // have left free.
            const int64_t N = 2, H = 256, W = 256, C = 2;

            expect_bit_identical_across_runs(
                [&](std::vector<uint32_t>& sink)
                {
                    auto in = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
                    fill_order_sensitive(in.data_as<float>(), in.num_elements);

                    auto out = engine->create_tensor({N, C}, Core::DataType::FLOAT32);
                    engine->reductions().mean_spatial(in, out);
                    engine->execute();

                    capture(out, sink);
                },
                N * C);
        }
    }
}
