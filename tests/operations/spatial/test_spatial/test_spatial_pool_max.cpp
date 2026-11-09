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
 * @file test_spatial_pool_max.cpp
 * @brief Tests SpatialOps::pool_max and pool_max_backward.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a rank-4 NHWC float tensor holding values in memory order. */
            Tensor pool_max_image(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c,
                                  const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor({n, h, w, c}, Core::DataType::Float32);
                fill_tensor(t, values);
                return t;
            }

            /** @brief Returns a rank-4 NHWC int64 tensor holding the indices in memory order. */
            Tensor pool_max_indices(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c,
                                    const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor({n, h, w, c}, Core::DataType::Int64);
                fill_tensor(t, values);
                return t;
            }

            /**
             * @brief Returns the 1x4x4x1 image the hand-checked cases pool.
             * @see tests/operations/spatial/README.md
             */
            Tensor pool_max_four_by_four(Engine& engine)
            {
                return pool_max_image(engine, 1, 4, 4, 1, {1, 3, 2, 0,
                                                           4, 2, 1, 5,
                                                           0, 7, 6, 2,
                                                           8, 1, 3, 9});
            }

            /** @brief Returns a 2x2 stride-2 geometry. */
            Spatial::ConvGeometry pool_max_tiles()
            {
                Spatial::ConvGeometry g;
                g.kernel_h = 2;
                g.kernel_w = 2;
                g.stride_h = 2;
                g.stride_w = 2;
                return g;
            }

            /** @brief Returns a 2x2 stride-1 geometry. */
            Spatial::ConvGeometry pool_max_overlapping()
            {
                Spatial::ConvGeometry g;
                g.kernel_h = 2;
                g.kernel_w = 2;
                return g;
            }

            /** @brief Returns count values of a reproducible pattern in -11 .. 11. */
            std::vector<float> pool_max_pattern(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>((i * 37) % 23) - 11.0f;
                return values;
            }

            /** @brief Returns the int64 indices as floats, exact below 2^24. */
            std::vector<float> pool_max_as_floats(const std::vector<int64_t>& indices)
            {
                return std::vector<float>(indices.begin(), indices.end());
            }
        } // namespace

        /** @brief Pools windows on the host engine. */
        class Small_SpatialPoolMaxTest : public HostEngineTest
        {
        };

        /** @brief Pools windows on the device and on the reference path. */
        class Medium_SpatialPoolMaxTest : public SushiBLASTest
        {
        };

        /**
         * @brief Tests that a padded tap never wins a window when every element is negative.
         * @see tests/operations/spatial/README.md
         */
        TEST_F(Small_SpatialPoolMaxTest, PaddingNeverWinsAWindow)
        {
            // Window (0,0) has one tap inside the image. See tests/operations/spatial/README.md.
            Tensor in = pool_max_image(*engine, 1, 2, 2, 1, {-4, -3,
                                                             -2, -1});
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;
            Tensor out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            Tensor indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, g);
            ASSERT_TRUE(recording->run_succeeds());

            // The windows see element (0,0), then (0,0) and (0,1), then (0,0) and (1,0), then all.
            expect_tensor_equal(out, {-4, -3, -2, -1});
        }

        TEST_F(Small_SpatialPoolMaxTest, ThePoolTakesTheWindowMaximum)
        {
            Tensor in = pool_max_four_by_four(*engine);
            Tensor out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            Tensor indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, pool_max_tiles());
            ASSERT_TRUE(recording->run_succeeds());

            // max{1,3,4,2}, max{2,0,1,5}, max{0,7,8,1} and max{6,2,3,9}.
            expect_tensor_equal(out, {4, 5, 8, 9});
        }

        TEST_F(Small_SpatialPoolMaxTest, TheIndexNamesTheElementTheMaximumCameFrom)
        {
            // The index is the logical flat NHWC position. See tests/operations/spatial/README.md.
            Tensor in = pool_max_four_by_four(*engine);
            Tensor out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            Tensor indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, pool_max_tiles());
            ASSERT_TRUE(recording->run_succeeds());

            // The values 4, 5, 8 and 9 sit at (h=1, w=0), (1, 3), (3, 0) and (3, 3).
            expect_tensor_equal(indices, {4, 7, 12, 15});
        }

        TEST_F(Small_SpatialPoolMaxTest, ATieGoesToTheFirstElementInScanOrder)
        {
            // A tie goes to the first tap in scan order. See tests/operations/spatial/README.md.
            Tensor in = pool_max_image(*engine, 1, 2, 2, 1, {5, 5,
                                                             5, 5});
            Tensor out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
            Tensor indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, pool_max_tiles());
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(out, {5});
            expect_tensor_equal(indices, {0});
        }

        TEST_F(Small_SpatialPoolMaxTest, TheBackwardRoutesEachGradientToItsArgmaxAndNowhereElse)
        {
            // The indices are the ones the forward pool gives. See tests/operations/spatial/README.md.
            Tensor indices = pool_max_indices(*engine, 1, 2, 2, 1, {4, 7, 12, 15});
            Tensor dy = pool_max_image(*engine, 1, 2, 2, 1, {10, 20, 30, 40});
            Tensor dx = engine->create_tensor({1, 4, 4, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).pool_max_backward(dy, indices, dx, pool_max_tiles());
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, {0, 0, 0, 0,
                                     10, 0, 0, 20,
                                     0, 0, 0, 0,
                                     30, 0, 0, 40});
        }

        TEST_F(Small_SpatialPoolMaxTest, OverlappingWindowsSelectTheSharedArgmax)
        {
            // All four windows select the one dominant pixel. See the test README.
            Tensor in = pool_max_image(*engine, 1, 3, 3, 1, {0, 0, 0,
                                                             0, 9, 0,
                                                             0, 0, 0});
            Tensor out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            Tensor indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, pool_max_overlapping());
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(indices, {4, 4, 4, 4});
        }

        TEST_F(Small_SpatialPoolMaxTest, OverlappingWindowsAccumulateAtASharedArgmax)
        {
            // The four gradients land on the centre pixel and add. See the test README.
            Tensor indices = pool_max_indices(*engine, 1, 2, 2, 1, {4, 4, 4, 4});
            Tensor dy = pool_max_image(*engine, 1, 2, 2, 1, {1, 2, 3, 4});
            Tensor dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::Float32);

            SushiBLAS::SpatialOps(recorder()).pool_max_backward(dy, indices, dx,
                                                                pool_max_overlapping());
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dx, {0, 0, 0,
                                     0, 10, 0,
                                     0, 0, 0});
        }

        TEST_F(Medium_SpatialPoolMaxTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            const int64_t N = 2, H = 7, W = 5, C = 3;
            Tensor in = pool_max_image(*engine, N, H, W, C, pool_max_pattern(N * H * W * C));

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
            Tensor out_ref = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            Tensor idx_ref = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);
            Tensor out_dev = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Float32);
            Tensor idx_dev = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::Int64);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).pool_max(in, out_ref, idx_ref, g);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).pool_max(in, out_dev, idx_dev, g);
            recording->run().value();

            expect_tensor_equal(out_dev, read_tensor<float>(out_ref));
            expect_tensor_equal(idx_dev, pool_max_as_floats(read_tensor<int64_t>(idx_ref)));
        }
    } // namespace Test
} // namespace SushiBLAS
