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
 * @file test_spatial_determinism.cpp
 * @brief Tests that the spatial backward kernels give the bits of the fixed-order reference path.
 * @author Mustafa Garip
 * @see tests/operations/spatial/README.md
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the bit patterns of the floats; `std::bit_cast` is C++20. */
            std::vector<uint32_t> determinism_bits(const std::vector<float>& values)
            {
                std::vector<uint32_t> bits(values.size());
                std::memcpy(bits.data(), values.data(), values.size() * sizeof(uint32_t));
                return bits;
            }

            /**
             * @brief Returns count values whose sum is order-sensitive.
             * @see tests/operations/spatial/README.md
             */
            std::vector<float> order_sensitive_values(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                {
                    switch (i % 4)
                    {
                        case 0: values[static_cast<std::size_t>(i)] = 1e8f; break;
                        case 1: values[static_cast<std::size_t>(i)] = 1.0f; break;
                        case 2: values[static_cast<std::size_t>(i)] = -1e8f; break;
                        default: values[static_cast<std::size_t>(i)] = 1.0f; break;
                    }
                }
                return values;
            }

            /** @brief Returns a rank-4 NHWC float tensor holding values in memory order. */
            Tensor determinism_image(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c,
                                     const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor({n, h, w, c}, Core::DataType::Float32);
                fill_tensor(t, values);
                return t;
            }

            /** @brief Returns the 3 x 3 stride-1 window padded by one. */
            Spatial::ConvGeometry determinism_padded_window()
            {
                Spatial::ConvGeometry g;
                g.kernel_h = 3;
                g.kernel_w = 3;
                g.pad_top = 1;
                g.pad_left = 1;
                return g;
            }
        } // namespace

        /** @brief Compares each device kernel with the fixed-order host reference path. */
        class Medium_SpatialDeterminismTest : public SushiBLASTest
        {
            protected:
                /**
                 * @brief Records an operation under each forced mode and expects the same bits.
                 * @param out_dims The shape of the result.
                 * @param record Records the operation into the tensor it is given.
                 */
                void expect_device_matches_reference(std::initializer_list<int64_t> out_dims,
                                                     const std::function<void(Tensor&)>& record)
                {
                    Tensor from_reference = engine->create_tensor(out_dims, Core::DataType::Float32);
                    Tensor from_device = engine->create_tensor(out_dims, Core::DataType::Float32);

                    set_mode(ExecutionMode::ForceReference);
                    record(from_reference);
                    recording->run().value();

                    set_mode(ExecutionMode::ForceSYCL);
                    record(from_device);
                    recording->run().value();

                    EXPECT_EQ(determinism_bits(read_tensor<float>(from_device)),
                              determinism_bits(read_tensor<float>(from_reference)));
                }
        };

        TEST_F(Medium_SpatialDeterminismTest, Col2ImMatchesTheReferenceBits)
        {
            // 3x3, stride 1, padded by one: every interior pixel gathers nine terms.
            const int64_t N = 2, H = 6, W = 7, C = 3;
            const Spatial::ConvGeometry g = determinism_padded_window();
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);
            Tensor cols = engine->create_tensor({N * out_h * out_w, 3 * 3 * C}, Core::DataType::Float32);
            fill_tensor(cols, order_sensitive_values(cols.num_elements));

            expect_device_matches_reference({N, H, W, C}, [&](Tensor& dx)
            {
                SushiBLAS::SpatialOps(recorder()).col2im(cols, dx, g, out_h, out_w);
            });
        }

        TEST_F(Medium_SpatialDeterminismTest, MaxPoolBackwardMatchesTheReferenceBits)
        {
            // 2x2, stride 1: a pixel selected by several windows accumulates all their gradients.
            const int64_t N = 2, H = 6, W = 7, C = 3;
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 1);
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 1);

            // One dominant pixel per channel plane, so every window in reach selects it.
            std::vector<float> dominant(static_cast<std::size_t>(N * H * W * C), 0.0f);
            for (int64_t n = 0; n < N; ++n)
                for (int64_t c = 0; c < C; ++c)
                    dominant[static_cast<std::size_t>(((n * H + 2) * W + 3) * C + c)] = 1.0f;
            Tensor in = determinism_image(*engine, N, H, W, C, dominant);
            Tensor indices = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);
            Tensor pooled = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            SushiBLAS::SpatialOps(recorder()).pool_max(in, pooled, indices, g);
            recording->run().value();
            Tensor dy = determinism_image(*engine, N, out_h, out_w, C,
                                          order_sensitive_values(N * out_h * out_w * C));

            expect_device_matches_reference({N, H, W, C}, [&](Tensor& dx)
            {
                SushiBLAS::SpatialOps(recorder()).pool_max_backward(dy, indices, dx, g);
            });
        }

        TEST_F(Medium_SpatialDeterminismTest, AvgPoolBackwardMatchesTheReferenceBits)
        {
            // 3x3 stride 1: every interior pixel gathers nine shares.
            const int64_t N = 2, H = 6, W = 7, C = 3;
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 3, 1);
            Tensor dy = determinism_image(*engine, N, out_h, out_w, C,
                                          order_sensitive_values(N * out_h * out_w * C));

            expect_device_matches_reference({N, H, W, C}, [&](Tensor& dx)
            {
                SushiBLAS::SpatialOps(recorder()).pool_avg_backward(dy, dx, g);
            });
        }

        TEST_F(Medium_SpatialDeterminismTest, WindowDotBackwardMatchesTheReferenceBits)
        {
            // 3x3, stride 1, padded by one, unit taps: every interior pixel gathers nine terms.
            const int64_t N = 2, H = 6, W = 7, C = 3;
            const Spatial::ConvGeometry g = determinism_padded_window();
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);
            Tensor dy = determinism_image(*engine, N, out_h, out_w, C,
                                          order_sensitive_values(N * out_h * out_w * C));
            Tensor weights = engine->create_tensor({3, 3, C}, Core::DataType::Float32);
            fill_tensor(weights, std::vector<float>(static_cast<std::size_t>(weights.num_elements), 1.0f));

            expect_device_matches_reference({N, H, W, C}, [&](Tensor& dx)
            {
                SushiBLAS::SpatialOps(recorder()).window_dot_backward(dy, weights, dx, g);
            });
        }

        TEST_F(Medium_SpatialDeterminismTest, UpsampleBackwardMatchesTheReferenceBits)
        {
            // Scale 3 in both axes, so every input pixel gathers nine terms.
            const int64_t N = 2, H = 5, W = 4, C = 3, scale = 3;
            Tensor dy = determinism_image(*engine, N, H * scale, W * scale, C,
                                          order_sensitive_values(N * H * scale * W * scale * C));

            expect_device_matches_reference({N, H, W, C}, [&](Tensor& dx)
            {
                SushiBLAS::SpatialOps(recorder()).upsample_nearest_backward(dy, dx, scale, scale);
            });
        }

        TEST_F(Medium_SpatialDeterminismTest, WindowDotWeightsBackwardMatchesTheReferenceBits)
        {
            // 3x3, stride 1, padded by one, unit input: every tap folds up to 84 gradients.
            const int64_t N = 2, H = 6, W = 7, C = 3;
            const Spatial::ConvGeometry g = determinism_padded_window();
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);
            Tensor dy = determinism_image(*engine, N, out_h, out_w, C,
                                          order_sensitive_values(N * out_h * out_w * C));
            Tensor in = determinism_image(*engine, N, H, W, C,
                                          std::vector<float>(static_cast<std::size_t>(N * H * W * C), 1.0f));

            expect_device_matches_reference({3, 3, C}, [&](Tensor& dweights)
            {
                SushiBLAS::SpatialOps(recorder()).window_dot_weights_backward(dy, in, dweights, g);
            });
        }

        TEST_F(Medium_SpatialDeterminismTest, MeanSpatialMatchesTheReferenceBits)
        {
            // Few segments over a long extent, so the device folds through the halving tree.
            const int64_t N = 2, H = 256, W = 256, C = 2;
            Tensor in = determinism_image(*engine, N, H, W, C,
                                          order_sensitive_values(N * H * W * C));

            expect_device_matches_reference({N, C}, [&](Tensor& out)
            {
                SushiBLAS::ReductionOps(recorder()).mean_spatial(in, out);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
