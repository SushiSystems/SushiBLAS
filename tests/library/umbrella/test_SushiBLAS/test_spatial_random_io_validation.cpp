/**************************************************************************/
/* test_spatial_random_io_validation.cpp                                  */
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
 * @file test_spatial_random_io_validation.cpp
 * @brief Tests the device pooling extremes, the degenerate random fills and the IO round trip.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Seed of the Bernoulli edge-probability tests.
            constexpr uint64_t K_BERNOULLI_SEED = 20261006;

            /// @brief Number of Bernoulli draws checked per edge probability.
            constexpr int64_t K_BERNOULLI_DRAWS = 100000;

            /// @brief The value both bounds of the degenerate uniform test share.
            constexpr double K_DEGENERATE_BOUND = 2.5;

            /// @brief Decimal places to_string prints in the transposed-view test.
            constexpr int K_PRINT_PRECISION = 1;

            /** @brief Names one Bernoulli probability and the only value it can draw. */
            struct BernoulliRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The probability of a one. */
                double probability;

                /** @brief The value every draw takes. */
                float expected;
            };

            /** @brief Names one file format and the dimensions of the tensor loaded from it. */
            struct ShapeRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief Whether the file is the raw binary form; otherwise the .sushi form. */
                bool binary;

                /** @brief The dimensions of the tensor the file is loaded into. */
                std::vector<int64_t> dims;

                /** @brief The status the load must give: the .sushi header names the shape, the raw form has none. */
                IOStatus refusal;
            };

            /** @brief Names a shape row by its name. */
            std::string shape_row_name(const ::testing::TestParamInfo<ShapeRow>& info)
            {
                return info.param.name;
            }

            /** @brief Names a row by its name. */
            std::string bernoulli_row_name(const ::testing::TestParamInfo<BernoulliRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Supplies an engine and removes the files a test wrote. */
        class Medium_SpatialRandomIOValidationTest : public SushiBLASTest
        {
            protected:
                /** @brief Removes every file the test asked for, then tears the engine down. */
                void TearDown() override
                {
                    for (const std::string& path : paths_)
                        std::remove(path.c_str());

                    SushiBLASTest::TearDown();
                }

                /** @brief Returns a path in the test temp directory and records it for removal. */
                std::string temp_path(const std::string& name)
                {
                    paths_.push_back(::testing::TempDir() + "sushiblas_unit_validation_" + name);
                    return paths_.back();
                }

                /** @brief Returns a rank-4 NHWC FLOAT32 tensor holding @p values in memory order. */
                Tensor image(int64_t n, int64_t h, int64_t w, int64_t c,
                             const std::vector<float>& values)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (size_t i = 0; i < values.size(); ++i)
                        p[i] = values[i];
                    return t;
                }

                /** @brief Expects every element of a FLOAT32 tensor to equal value. */
                static void expect_every_element(const Tensor& t, float value)
                {
                    expect_tensor_equal(t, std::vector<float>(static_cast<size_t>(t.num_elements), value));
                }

            private:
                /// @brief Lists the files this test created.
                std::vector<std::string> paths_;
        };

        TEST_F(Medium_SpatialRandomIOValidationTest, PoolMaxPropagatesANaNTap)
        {
            constexpr int64_t K_NAN_INDEX = 1;

            const float nan = std::numeric_limits<float>::quiet_NaN();
            auto in = image(1, 1, 3, 1, {5.0f, nan, 9.0f});
            auto out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            Spatial::ConvGeometry g;
            g.kernel_w = 3;

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, g);
            recording->run().value();

            EXPECT_TRUE(std::isnan(out.data_as<float>()[0]));
            EXPECT_EQ(indices.data_as<int64_t>()[0], K_NAN_INDEX);
        }

        TEST_F(Medium_SpatialRandomIOValidationTest, PoolMaxOfAnAllNegativeInfinityWindowIsNegativeInfinity)
        {
            constexpr int64_t K_FIRST_TAP_INDEX = 0;

            const float neg_inf = -std::numeric_limits<float>::infinity();
            auto in = image(1, 1, 3, 1, {neg_inf, neg_inf, neg_inf});
            auto out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            Spatial::ConvGeometry g;
            g.kernel_w = 3;

            SushiBLAS::SpatialOps(recorder()).pool_max(in, out, indices, g);
            recording->run().value();

            EXPECT_EQ(out.data_as<float>()[0], neg_inf);
            EXPECT_EQ(indices.data_as<int64_t>()[0], K_FIRST_TAP_INDEX);
        }

        /// @brief Draws a Bernoulli fill at one edge probability per row.
        class Medium_BernoulliEdge : public Medium_SpatialRandomIOValidationTest,
                                     public ::testing::WithParamInterface<BernoulliRow> {};

        /// @brief Tests that a Bernoulli fill at probability 0 or 1 draws only that value.
        TEST_P(Medium_BernoulliEdge, FillsOnlyTheValueTheProbabilityNames)
        {
            auto t = engine->create_tensor({K_BERNOULLI_DRAWS}, Core::DataType::Float32);

            SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(K_BERNOULLI_SEED);
            SushiBLAS::RandomOps(recorder(), engine->rng()).bernoulli(t, GetParam().probability);
            recording->run().value();

            expect_every_element(t, GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            Probabilities, Medium_BernoulliEdge,
            ::testing::Values(BernoulliRow{"Zero", 0.0, 0.0f}, BernoulliRow{"One", 1.0, 1.0f}),
            bernoulli_row_name);

        TEST_F(Medium_SpatialRandomIOValidationTest, UniformWithEqualBoundsFillsThatBound)
        {
            auto t = engine->create_tensor({8}, Core::DataType::Float32);

            SushiBLAS::RandomOps(recorder(), engine->rng())
                .uniform(t, K_DEGENERATE_BOUND, K_DEGENERATE_BOUND);
            recording->run().value();

            expect_every_element(t, static_cast<float>(K_DEGENERATE_BOUND));
        }

        TEST_F(Medium_SpatialRandomIOValidationTest, ToStringReadsATransposedViewThroughItsStrides)
        {
            // Row-major 2x3 holds 0..5; its transpose (i, j) reads memory i + 3j.
            auto source = create_initialized_tensor<float>({2, 3},
                                                           {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
            const Tensor view = source.transpose(0, 1);

            const std::string text = SushiBLAS::IO().to_string(view, K_PRINT_PRECISION);

            EXPECT_NE(text.find("[0.0, 3.0]"), std::string::npos) << text;
            EXPECT_NE(text.find("[1.0, 4.0]"), std::string::npos) << text;
            EXPECT_NE(text.find("[2.0, 5.0]"), std::string::npos) << text;
        }

        TEST_F(Medium_SpatialRandomIOValidationTest, SaveThenLoadReturnsTheOriginalValues)
        {
            const std::vector<float> values{1.5f, -2.25f, 3.0f, 4.75f};
            const std::string path = temp_path("round_trip.sushi");

            auto source = create_initialized_tensor<float>({2, 2}, values);
            ASSERT_EQ(SushiBLAS::IO().save(source, path), SushiBLAS::IOStatus::Ok);

            auto target = engine->create_tensor({2, 2}, Core::DataType::Float32);
            ASSERT_EQ(SushiBLAS::IO().load(target, path), SushiBLAS::IOStatus::Ok);

            expect_tensor_equal(target, values);
        }

        TEST_F(Medium_SpatialRandomIOValidationTest, LoadFromAMissingPathIsUnreadable)
        {
            auto target = engine->create_tensor({2, 2}, Core::DataType::Float32);

            EXPECT_EQ(SushiBLAS::IO().load(target, temp_path("missing.sushi")),
                      SushiBLAS::IOStatus::FileUnreadable);
        }

        TEST_F(Medium_SpatialRandomIOValidationTest, SaveToAnUnwritablePathIsUnwritableAndLeavesNoFile)
        {
            const std::string path = temp_path("no_such_folder/unwritable.sushi");
            auto source = create_initialized_tensor<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

            EXPECT_EQ(SushiBLAS::IO().save(source, path), SushiBLAS::IOStatus::FileUnwritable);
            EXPECT_FALSE(std::ifstream(path).good()) << "the partial file must be removed";
        }

        /**
         * @brief Loads a saved 2 x 2 file into a tensor of another shape, one format and shape per row.
         *
         * A .sushi header holds the shape, so a rank or a dimension that differs is a ShapeMismatch.
         * A raw file holds the 16 bytes alone and knows no rank: a tensor of 24 bytes finds the file
         * short (Truncated), and a rank of the same 16 bytes would load, so it has no row.
         */
        class Medium_LoadShape : public Medium_SpatialRandomIOValidationTest,
                                 public ::testing::WithParamInterface<ShapeRow> {};

        TEST_P(Medium_LoadShape, OfAnotherRankOrDimensionIsRefused)
        {
            const std::string path = temp_path(GetParam().binary ? "shape.bin" : "shape.sushi");
            auto source = create_initialized_tensor<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
            auto target = engine->create_tensor(
                SushiRuntime::span<const int64_t>(GetParam().dims.data(), GetParam().dims.size()),
                Core::DataType::Float32);

            const IOStatus saved = GetParam().binary ? SushiBLAS::IO().save_bin(source, path)
                                                      : SushiBLAS::IO().save(source, path);
            ASSERT_EQ(saved, IOStatus::Ok);

            const IOStatus loaded = GetParam().binary ? SushiBLAS::IO().load_bin(target, path)
                                                       : SushiBLAS::IO().load(target, path);
            EXPECT_EQ(loaded, GetParam().refusal);
        }

        INSTANTIATE_TEST_SUITE_P(
            Formats, Medium_LoadShape,
            ::testing::Values(
                ShapeRow{"SushiAnotherDimension", false, {2, 3}, IOStatus::ShapeMismatch},
                ShapeRow{"SushiAnotherRank", false, {4}, IOStatus::ShapeMismatch},
                ShapeRow{"BinAnotherDimension", true, {2, 3}, IOStatus::Truncated}),
            shape_row_name);
    } // namespace Test
} // namespace SushiBLAS
