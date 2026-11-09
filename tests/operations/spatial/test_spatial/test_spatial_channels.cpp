/**************************************************************************/
/* test_spatial_channels.cpp                                              */
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
 * @file test_spatial_channels.cpp
 * @brief Tests SpatialOps::concat_channels and slice_channels.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
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
            /** @brief Returns a rank-4 NHWC float tensor holding values in memory order. */
            Tensor image(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c,
                         const std::vector<float>& values)
            {
                Tensor t = engine.create_tensor({n, h, w, c}, Core::DataType::Float32);
                fill_tensor(t, values);
                return t;
            }

            /** @brief Returns a rank-4 NHWC float tensor whose every element is value. */
            Tensor filled(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c, float value)
            {
                return image(engine, n, h, w, c,
                             std::vector<float>(static_cast<std::size_t>(n * h * w * c), value));
            }

            /** @brief Returns count values of a reproducible pattern that depends on seed. */
            std::vector<float> patterned_values(int64_t count, int64_t seed)
            {
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] =
                        static_cast<float>(((i + seed) * 37) % 23) - 11.0f;
                return values;
            }

            /** @brief Returns a rank-4 NHWC float tensor filled with patterned_values. */
            Tensor patterned(Engine& engine, int64_t n, int64_t h, int64_t w, int64_t c,
                             int64_t seed)
            {
                return image(engine, n, h, w, c, patterned_values(n * h * w * c, seed));
            }

            /** @brief Expects record to throw std::invalid_argument whose message holds fragment. */
            void expect_refusal_naming(const std::function<void()>& record,
                                       const std::string& fragment)
            {
                try
                {
                    record();
                    ADD_FAILURE() << "the call accepted the malformed operands";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string message = error.what();
                    EXPECT_NE(message.find(fragment), std::string::npos) << message;
                }
            }

            /** @brief Names a malformed call and the operation and operand its refusal names. */
            struct MalformedRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief Records the malformed call. */
                std::function<void(SpatialOps&, Engine&)> record;

                /** @brief The operation the message names. */
                std::string op;

                /** @brief The operand the message names. */
                std::string operand;
            };

            /** @brief Returns the 2 x 2 stride-2 window of the malformed pooling rows. */
            Spatial::ConvGeometry pooling_window()
            {
                Spatial::ConvGeometry g;
                g.kernel_h = 2;
                g.kernel_w = 2;
                g.stride_h = 2;
                g.stride_w = 2;
                return g;
            }

            /** @brief Returns the rows of the malformed operands. */
            std::vector<MalformedRow> malformed_rows()
            {
                return {
                    {"PoolAvgInputOfRankThree",
                     [](SpatialOps& spatial, Engine& e)
                     {
                         Tensor rank3 = e.create_tensor({1, 4, 4}, Core::DataType::Float32);
                         Tensor pooled = filled(e, 1, 2, 2, 1, 0.0f);
                         spatial.pool_avg(rank3, pooled, pooling_window());
                     },
                     "spatial.pool_avg", "in"},
                    {"Im2ColColumnsOfAnotherDtype",
                     [](SpatialOps& spatial, Engine& e)
                     {
                         Tensor picture = filled(e, 1, 4, 4, 1, 1.0f);
                         Tensor wide_cols = e.create_tensor({4, 4}, Core::DataType::Float64);
                         spatial.im2col(picture, wide_cols, pooling_window(), 2, 2);
                     },
                     "spatial.im2col", "cols"},
                    {"PoolMaxIndicesOfAnotherDtype",
                     [](SpatialOps& spatial, Engine& e)
                     {
                         Tensor picture = filled(e, 1, 4, 4, 1, 1.0f);
                         Tensor pooled = filled(e, 1, 2, 2, 1, 0.0f);
                         Tensor narrow = e.create_tensor({1, 2, 2, 1}, Core::DataType::Int32);
                         spatial.pool_max(picture, pooled, narrow, pooling_window());
                     },
                     "spatial.pool_max", "indices"},
                    {"ConcatAtANegativeOffset",
                     [](SpatialOps& spatial, Engine& e)
                     {
                         Tensor three = filled(e, 1, 2, 2, 3, 1.0f);
                         Tensor five = filled(e, 1, 2, 2, 5, 0.0f);
                         spatial.concat_channels(three, five, -1);
                     },
                     "spatial.concat_channels", "dst"}};
            }

            /** @brief Names a row by its name. */
            std::string malformed_row_name(const ::testing::TestParamInfo<MalformedRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Joins and slices channels on the host engine. */
        class Small_SpatialChannelsTest : public HostEngineTest
        {
        };

        /** @brief Records one malformed call per row on a host engine. */
        class Small_SpatialChannelsMalformedTest
            : public HostEngineTest,
              public ::testing::WithParamInterface<MalformedRow>
        {
        };

        /** @brief Joins and slices channels on the device and on the reference path. */
        class Medium_SpatialChannelsTest : public SushiBLASTest
        {
        };

        TEST_F(Small_SpatialChannelsTest, ConcatPlacesEachSourceAtItsOffset)
        {
            // Two pixels expose an ignored column stride. See tests/operations/spatial/README.md.
            Tensor a = image(*engine, 1, 1, 2, 2, {10, 11,
                                                   12, 13});
            Tensor b = image(*engine, 1, 1, 2, 3, {20, 21, 22,
                                                    23, 24, 25});
            const float sentinel = -99.0f;
            Tensor dst = filled(*engine, 1, 1, 2, 5, sentinel);

            SushiBLAS::SpatialOps(recorder()).concat_channels(a, dst, 0);
            ASSERT_TRUE(recording->run_succeeds());

            // A's write must stop at channel 2. The sentinel is what says so.
            expect_tensor_equal(dst, {10, 11, sentinel, sentinel, sentinel,
                                      12, 13, sentinel, sentinel, sentinel});

            SushiBLAS::SpatialOps(recorder()).concat_channels(b, dst, 2);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dst, {10, 11, 20, 21, 22,
                                      12, 13, 23, 24, 25});
        }

        TEST_F(Small_SpatialChannelsTest, SliceIsConcatsInverse)
        {
            Tensor a = patterned(*engine, 2, 3, 4, 2, 0);
            Tensor b = patterned(*engine, 2, 3, 4, 5, 7);
            Tensor joined = filled(*engine, 2, 3, 4, 7, -1.0f);
            Tensor a_back = filled(*engine, 2, 3, 4, 2, 0.0f);
            Tensor b_back = filled(*engine, 2, 3, 4, 5, 0.0f);

            SushiBLAS::SpatialOps(recorder()).concat_channels(a, joined, 0);
            SushiBLAS::SpatialOps(recorder()).concat_channels(b, joined, 2);
            SushiBLAS::SpatialOps(recorder()).slice_channels(joined, 0, a_back);
            SushiBLAS::SpatialOps(recorder()).slice_channels(joined, 2, b_back);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(a_back, patterned_values(a.num_elements, 0));
            expect_tensor_equal(b_back, patterned_values(b.num_elements, 7));
        }

        TEST_F(Small_SpatialChannelsTest, AConcatPastTheDestinationsChannelCountIsRefused)
        {
            // Writing three channels starting at 3 needs six, and there are five.
            Tensor three = filled(*engine, 1, 2, 2, 3, 1.0f);
            Tensor five = filled(*engine, 1, 2, 2, 5, 0.0f);

            expect_refusal_naming(
                [&] { SushiBLAS::SpatialOps(recorder()).concat_channels(three, five, 3); },
                "spatial.concat_channels");
        }

        TEST_F(Small_SpatialChannelsTest, ASlicePastTheSourcesChannelCountIsRefused)
        {
            // Reading three channels starting at 4 needs seven, and there are five.
            Tensor five = filled(*engine, 1, 2, 2, 5, 0.0f);
            Tensor destination = filled(*engine, 1, 2, 2, 3, 0.0f);

            expect_refusal_naming(
                [&] { SushiBLAS::SpatialOps(recorder()).slice_channels(five, 4, destination); },
                "spatial.slice_channels");
        }

        TEST_F(Small_SpatialChannelsTest, AStridedSourceIsAddressedThroughItsStrides)
        {
            // The view sees only the parent's even channels. See the test README.
            Tensor parent = image(*engine, 1, 2, 2, 4, {0, 1, 2, 3,      4, 5, 6, 7,
                                                        8, 9, 10, 11,    12, 13, 14, 15});

            const std::vector<int64_t> dims = {1, 2, 2, 2};
            std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
            strides[0] = 16;
            strides[1] = 8;
            strides[2] = 4;
            strides[3] = 2;

            const Tensor view(parent.storage, dims, strides, Core::DataType::Float32,
                              parent.storage_offset, parent.layout);
            Tensor dst = filled(*engine, 1, 2, 2, 2, -1.0f);

            SushiBLAS::SpatialOps(recorder()).concat_channels(view, dst, 0);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(dst, {0, 2,   4, 6,
                                      8, 10,  12, 14});
        }

        TEST_P(Small_SpatialChannelsMalformedTest, TheRefusalNamesTheOperationAndTheOperand)
        {
            const MalformedRow& row = GetParam();
            SushiBLAS::SpatialOps spatial(recorder());

            expect_refusal_naming([&] { row.record(spatial, *engine); }, row.op);
            expect_refusal_naming([&] { row.record(spatial, *engine); },
                                  "operand '" + row.operand + "'");
        }

        INSTANTIATE_TEST_SUITE_P(Operands, Small_SpatialChannelsMalformedTest,
                                 ::testing::ValuesIn(malformed_rows()), malformed_row_name);

        TEST_F(Medium_SpatialChannelsTest, TheSYCLConcatAgreesWithTheReferenceConcat)
        {
            // The two forced modes are compared here. See tests/operations/spatial/README.md.
            const int64_t N = 2, H = 3, W = 4, C = 3, WIDE = 7, OFFSET = 2;
            Tensor src = patterned(*engine, N, H, W, C, 5);
            Tensor from_reference = patterned(*engine, N, H, W, WIDE, 13);
            Tensor from_device = patterned(*engine, N, H, W, WIDE, 13);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).concat_channels(src, from_reference, OFFSET);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).concat_channels(src, from_device, OFFSET);
            recording->run().value();

            expect_tensor_equal(from_device, read_tensor<float>(from_reference));
        }

        TEST_F(Medium_SpatialChannelsTest, TheSYCLSliceAgreesWithTheReferenceSlice)
        {
            const int64_t N = 2, H = 3, W = 4, C = 3, WIDE = 7, OFFSET = 2;
            Tensor joined = patterned(*engine, N, H, W, WIDE, 13);
            Tensor sliced_reference = filled(*engine, N, H, W, C, 0.0f);
            Tensor sliced_device = filled(*engine, N, H, W, C, 0.0f);

            set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(recorder()).slice_channels(joined, OFFSET, sliced_reference);
            recording->run().value();

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::SpatialOps(recorder()).slice_channels(joined, OFFSET, sliced_device);
            recording->run().value();

            expect_tensor_equal(sliced_device, read_tensor<float>(sliced_reference));
        }
    } // namespace Test
} // namespace SushiBLAS
