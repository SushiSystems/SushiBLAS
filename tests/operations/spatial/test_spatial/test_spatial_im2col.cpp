/**************************************************************************/
/* test_spatial_im2col.cpp                                                */
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
 * @file test_spatial_im2col.cpp
 * @brief Tests SpatialOps::im2col's gather order, padding, dilation and refusals.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
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
            /** @brief Returns count values 0, 1, 2 and so on. */
            std::vector<float> im2col_ramp_values(int64_t count)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
                return values;
            }

            /** @brief Returns an NHWC tensor holding 0, 1, 2, ... in memory order. */
            Tensor im2col_ramp(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c)
            {
                Tensor t = engine.create_tensor({n, h, w, c}, Core::DataType::Float32);
                fill_tensor(t, im2col_ramp_values(t.num_elements));
                return t;
            }

            /** @brief Expects record to throw std::invalid_argument with exactly this message. */
            void expect_im2col_refusal(const std::function<void()>& record,
                                       const std::string& message)
            {
                try
                {
                    record();
                    ADD_FAILURE() << "the call accepted the malformed operands";
                }
                catch (const std::invalid_argument& error)
                {
                    EXPECT_EQ(std::string(error.what()), message);
                }
            }
        } // namespace

        /** @brief Gathers windows on the host engine. */
        class Small_SpatialIm2ColTest : public HostEngineTest
        {
        };

        /** @brief Gathers windows on the device and on the reference path. */
        class Medium_SpatialIm2ColTest : public SushiBLASTest
        {
        };

        TEST_F(Small_SpatialIm2ColTest, AUnitWindowIsTheIdentityGather)
        {
            // A 1x1 window's columns are the input reshaped. See the test README.
            Tensor in = im2col_ramp(*engine, 2, 3, 4, 5);
            Tensor cols = engine->create_tensor({2 * 3 * 4, 5}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            SushiBLAS::SpatialOps(recorder()).im2col(in, cols, g);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(cols, im2col_ramp_values(in.num_elements));
        }

        TEST_F(Small_SpatialIm2ColTest, EachRowIsOneWindowWithChannelInnermost)
        {
            // 2x2 stride 1 gives 4 windows of 8 entries each. See the test README.
            Tensor in = im2col_ramp(*engine, 1, 3, 3, 2);
            Tensor cols = engine->create_tensor({4, 8}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            SushiBLAS::SpatialOps(recorder()).im2col(in, cols, g);
            ASSERT_TRUE(recording->run_succeeds());

            // Element (h, w, c) holds (h*3 + w)*2 + c. Row k is the window at (k / 2, k % 2).
            expect_tensor_equal(cols, {0, 1, 2, 3, 6, 7, 8, 9,
                                       2, 3, 4, 5, 8, 9, 10, 11,
                                       6, 7, 8, 9, 12, 13, 14, 15,
                                       8, 9, 10, 11, 14, 15, 16, 17});
        }

        TEST_F(Small_SpatialIm2ColTest, OutOfBoundsTapsReadZeroRatherThanNeighbouringMemory)
        {
            // Padding is a parameter, not a padded copy. See tests/operations/spatial/README.md.
            Tensor in = im2col_ramp(*engine, 1, 2, 2, 1);
            Tensor cols = engine->create_tensor({4, 9}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            SushiBLAS::SpatialOps(recorder()).im2col(in, cols, g);
            ASSERT_TRUE(recording->run_succeeds());

            // Row k is the 3 x 3 window centred on pixel (k / 2, k % 2) of the 2 x 2 image 0 1 / 2 3.
            expect_tensor_equal(cols, {0, 0, 0, 0, 0, 1, 0, 2, 3,
                                       0, 0, 0, 0, 1, 0, 2, 3, 0,
                                       0, 0, 1, 0, 2, 3, 0, 0, 0,
                                       0, 1, 0, 2, 3, 0, 0, 0, 0});
        }

        TEST_F(Small_SpatialIm2ColTest, DilationSpacesTheTapsWithoutMovingTheWindows)
        {
            Tensor in = im2col_ramp(*engine, 1, 5, 5, 1);
            Tensor cols = engine->create_tensor({1, 9}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.dilation_h = 2;
            g.dilation_w = 2;

            SushiBLAS::SpatialOps(recorder()).im2col(in, cols, g);
            ASSERT_TRUE(recording->run_succeeds());

            // The single window's taps are rows {0,2,4} x columns {0,2,4}.
            expect_tensor_equal(cols, {0, 2, 4, 10, 12, 14, 20, 22, 24});
        }

        TEST_F(Small_SpatialIm2ColTest, TheTwoOperandOverloadRefusesAnAsymmetricGeometry)
        {
            // The two-operand overload has no pad_right. See tests/operations/spatial/README.md.
            Tensor in = im2col_ramp(*engine, 1, 3, 3, 1);
            Tensor cols = engine->create_tensor({9, 4}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            // Symmetric-padding derivation gives W_o = 2, which does not divide 9.
            expect_im2col_refusal([&] { SushiBLAS::SpatialOps(recorder()).im2col(in, cols, g); },
                                  "spatial.im2col: operand 'cols' asks for 9 windows per image, "
                                  "which 2 output columns do not divide");
        }

        TEST_F(Small_SpatialIm2ColTest, TheExplicitExtentOverloadGathersAnAsymmetricGeometry)
        {
            Tensor in = im2col_ramp(*engine, 1, 3, 3, 1);
            Tensor cols = engine->create_tensor({9, 4}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            SushiBLAS::SpatialOps(recorder()).im2col(in, cols, g, 3, 3);
            ASSERT_TRUE(recording->run_succeeds());

            // Window (ho, wo) takes pixels (ho, wo) to (ho + 1, wo + 1); a tap past the edge reads zero.
            expect_tensor_equal(cols, {0, 1, 3, 4,
                                       1, 2, 4, 5,
                                       2, 0, 5, 0,
                                       3, 4, 6, 7,
                                       4, 5, 7, 8,
                                       5, 0, 8, 0,
                                       6, 7, 0, 0,
                                       7, 8, 0, 0,
                                       8, 0, 0, 0});
        }

        TEST_F(Small_SpatialIm2ColTest, AColumnMatrixOfTheWrongShapeIsRefusedByName)
        {
            Tensor in = im2col_ramp(*engine, 1, 4, 4, 2);
            Tensor cols = engine->create_tensor({9, 7}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            // 7 != 2 * 2 * 2, the kernel area times the channels.
            expect_im2col_refusal(
                [&] { SushiBLAS::SpatialOps(recorder()).im2col(in, cols, g, 3, 3); },
                "spatial.im2col: operand 'cols' has 7 columns, expected 8");
        }

        TEST_F(Small_SpatialIm2ColTest, ARankThatIsNotFourIsRefusedAtRecordTime)
        {
            const std::vector<int64_t> dims = {4, 4, 2};
            const Tensor in = Tensor::make_shape_only(
                SushiRuntime::span<const int64_t>(dims.data(), dims.size()));
            Tensor cols = engine->create_tensor({16, 8}, Core::DataType::Float32);

            expect_im2col_refusal(
                [&] { SushiBLAS::SpatialOps(recorder()).im2col(in, cols, Spatial::ConvGeometry{}); },
                "spatial.im2col: operand 'in' has rank 3, expected 4");
        }

        TEST_F(Medium_SpatialIm2ColTest, TheSYCLKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            Tensor in = im2col_ramp(*engine, 2, 7, 5, 3);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 1;
            g.dilation_h = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(7, 1, 1, 2, 3, 2);
            const int64_t out_w = Spatial::output_extent(5, 1, 1, 1, 2, 1);
            const int64_t rows = 2 * out_h * out_w;
            const int64_t width = 3 * 2 * 3;

            Tensor from_reference = engine->create_tensor({rows, width}, Core::DataType::Float32);
            Tensor from_device = engine->create_tensor({rows, width}, Core::DataType::Float32);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).im2col(in, from_reference, g, out_h, out_w);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).im2col(in, from_device, g, out_h, out_w);
            recording->run().value();

            expect_tensor_equal(from_device, read_tensor<float>(from_reference));
        }
    } // namespace Test
} // namespace SushiBLAS
