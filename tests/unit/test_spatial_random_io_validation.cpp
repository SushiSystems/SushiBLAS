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
 * @file test_spatial_random_io_validation.cpp
 * @brief Tests the record-time checks of spatial geometry, random fills and IO.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Seed of the uniform draw test.
            constexpr uint64_t K_UNIFORM_SEED = 20261005;
            /// @brief Number of uniform draws checked against the upper bound.
            constexpr int64_t K_UNIFORM_DRAWS = 100000;
            /// @brief A Poisson mean above K_MAX_POISSON_LAMBDA.
            constexpr double K_LAMBDA_ABOVE_CAP = 2.0e7;
            /// @brief A negative Poisson mean.
            constexpr double K_LAMBDA_NEGATIVE = -1.0;
            /// @brief Seed of the Bernoulli edge-probability tests.
            constexpr uint64_t K_BERNOULLI_SEED = 20261006;
            /// @brief Number of Bernoulli draws checked per edge probability.
            constexpr int64_t K_BERNOULLI_DRAWS = 100000;
            /// @brief The bound FLOAT32 discrete_uniform accepts in magnitude, 2^24.
            constexpr int32_t K_FLOAT32_EXACT_LIMIT = 16777216;
            /// @brief A float bound whose doubled magnitude overflows FLOAT32.
            constexpr double K_LARGE_FLOAT_BOUND = 3.0e38;
            /// @brief A double bound whose doubled magnitude overflows FLOAT64.
            constexpr double K_LARGE_DOUBLE_BOUND = 1.0e308;
            /// @brief The value both bounds of the degenerate uniform test share.
            constexpr double K_DEGENERATE_BOUND = 2.5;
            /// @brief Decimal places to_string prints in the transposed-view test.
            constexpr int K_PRINT_PRECISION = 1;
        } // namespace

        /** @brief Supplies an engine and removes the files a test wrote. */
        class Unit_SpatialRandomIOValidationTest : public SushiBLASTest
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

            private:
                /// @brief Lists the files this test created.
                std::vector<std::string> paths_;
        };

        TEST_F(Unit_SpatialRandomIOValidationTest, GeometryWithZeroStrideIsRefused)
        {
            auto in = image(1, 2, 2, 1, {1.0f, 2.0f, 3.0f, 4.0f});
            auto out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 0;

            EXPECT_THROW(
                SushiBLAS::SpatialOps(*engine).pool_avg(in, out, g),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, PoolingWindowMadeOnlyOfPaddingIsRefused)
        {
            // Window 0 covers rows -2 and -1 of a 2-row input: both taps are padding.
            auto in = image(1, 2, 2, 1, {1.0f, 2.0f, 3.0f, 4.0f});
            auto out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 2;

            EXPECT_THROW(
                SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, g),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, PoolMaxRefusesAStridedIndexTensor)
        {
            auto in = image(1, 2, 2, 1, {1.0f, 2.0f, 3.0f, 4.0f});
            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            auto dense = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);
            auto indices = dense.transpose(1, 2);

            EXPECT_THROW(
                SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, Spatial::ConvGeometry{}),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, PoolMaxPropagatesANaNTap)
        {
            constexpr int64_t K_NAN_INDEX = 1;

            const float nan = std::numeric_limits<float>::quiet_NaN();
            auto in = image(1, 1, 3, 1, {5.0f, nan, 9.0f});
            auto out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            Spatial::ConvGeometry g;
            g.kernel_w = 3;

            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, g);
            engine->execute();

            EXPECT_TRUE(std::isnan(out.data_as<float>()[0]));
            EXPECT_EQ(indices.data_as<int64_t>()[0], K_NAN_INDEX);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, UniformNeverReturnsItsUpperBound)
        {
            auto t = engine->create_tensor({K_UNIFORM_DRAWS}, Core::DataType::Float32);

            SushiBLAS::RandomOps(*engine).set_seed(K_UNIFORM_SEED);
            SushiBLAS::RandomOps(*engine).uniform(t, 0.0, 1.0);
            engine->execute();

            const float* p = t.data_as<float>();
            for (int64_t i = 0; i < K_UNIFORM_DRAWS; ++i)
            {
                ASSERT_GE(p[i], 0.0f) << "at index " << i;
                ASSERT_LT(p[i], 1.0f) << "at index " << i;
            }
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, PoissonRefusesALambdaAboveTheCap)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float32);

            EXPECT_THROW(
                SushiBLAS::RandomOps(*engine).poisson(t, K_LAMBDA_ABOVE_CAP),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, PoissonRefusesANegativeLambda)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float32);

            EXPECT_THROW(
                SushiBLAS::RandomOps(*engine).poisson(t, K_LAMBDA_NEGATIVE),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, PoolMaxBackwardRefusesATransposedOutputGradient)
        {
            auto dense = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
            const Tensor dy = dense.transpose(1, 2);
            auto indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int64);
            auto dx = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);

            EXPECT_THROW(
                SushiBLAS::SpatialOps(*engine).pool_max_backward(
                    dy, indices, dx, Spatial::ConvGeometry{}),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, PoolMaxOfAnAllNegativeInfinityWindowIsNegativeInfinity)
        {
            constexpr int64_t K_FIRST_TAP_INDEX = 0;

            const float neg_inf = -std::numeric_limits<float>::infinity();
            auto in = image(1, 1, 3, 1, {neg_inf, neg_inf, neg_inf});
            auto out = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Float32);
            auto indices = engine->create_tensor({1, 1, 1, 1}, Core::DataType::Int64);

            Spatial::ConvGeometry g;
            g.kernel_w = 3;

            SushiBLAS::SpatialOps(*engine).pool_max(in, out, indices, g);
            engine->execute();

            EXPECT_EQ(out.data_as<float>()[0], neg_inf);
            EXPECT_EQ(indices.data_as<int64_t>()[0], K_FIRST_TAP_INDEX);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, BernoulliWithZeroProbabilityFillsZeros)
        {
            auto t = engine->create_tensor({K_BERNOULLI_DRAWS}, Core::DataType::Float32);

            SushiBLAS::RandomOps(*engine).set_seed(K_BERNOULLI_SEED);
            SushiBLAS::RandomOps(*engine).bernoulli(t, 0.0);
            engine->execute();

            const float* p = t.data_as<float>();
            for (int64_t i = 0; i < K_BERNOULLI_DRAWS; ++i)
                ASSERT_EQ(p[i], 0.0f) << "at index " << i;
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, BernoulliWithUnitProbabilityFillsOnes)
        {
            auto t = engine->create_tensor({K_BERNOULLI_DRAWS}, Core::DataType::Float32);

            SushiBLAS::RandomOps(*engine).set_seed(K_BERNOULLI_SEED);
            SushiBLAS::RandomOps(*engine).bernoulli(t, 1.0);
            engine->execute();

            const float* p = t.data_as<float>();
            for (int64_t i = 0; i < K_BERNOULLI_DRAWS; ++i)
                ASSERT_EQ(p[i], 1.0f) << "at index " << i;
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, UniformRefusesBoundsOutOfOrder)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float32);

            EXPECT_THROW(SushiBLAS::RandomOps(*engine).uniform(t, 1.0, 0.0), std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, UniformRefusesANonFiniteBound)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float64);
            const double inf = std::numeric_limits<double>::infinity();
            const double nan = std::numeric_limits<double>::quiet_NaN();

            EXPECT_THROW(SushiBLAS::RandomOps(*engine).uniform(t, 0.0, inf), std::invalid_argument);
            EXPECT_THROW(SushiBLAS::RandomOps(*engine).uniform(t, nan, 1.0), std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, UniformRefusesAWidthThatOverflowsTheDtype)
        {
            auto single = engine->create_tensor({4}, Core::DataType::Float32);
            auto wide = engine->create_tensor({4}, Core::DataType::Float64);

            EXPECT_THROW(
                SushiBLAS::RandomOps(*engine).uniform(
                    single, -K_LARGE_FLOAT_BOUND, K_LARGE_FLOAT_BOUND),
                std::invalid_argument);
            EXPECT_THROW(
                SushiBLAS::RandomOps(*engine).uniform(
                    wide, -K_LARGE_DOUBLE_BOUND, K_LARGE_DOUBLE_BOUND),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, UniformWithEqualBoundsFillsThatBound)
        {
            auto t = engine->create_tensor({8}, Core::DataType::Float32);

            SushiBLAS::RandomOps(*engine).uniform(t, K_DEGENERATE_BOUND, K_DEGENERATE_BOUND);
            engine->execute();

            const float* p = t.data_as<float>();
            for (int64_t i = 0; i < t.num_elements; ++i)
                ASSERT_EQ(p[i], static_cast<float>(K_DEGENERATE_BOUND)) << "at index " << i;
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, DiscreteUniformIntoFloat32RefusesBoundsPastTwoToTheTwentyFour)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float32);

            EXPECT_NO_THROW(
                SushiBLAS::RandomOps(*engine).discrete_uniform(
                    t, -K_FLOAT32_EXACT_LIMIT, K_FLOAT32_EXACT_LIMIT));
            EXPECT_THROW(
                SushiBLAS::RandomOps(*engine).discrete_uniform(t, 0, K_FLOAT32_EXACT_LIMIT + 1),
                std::invalid_argument);
            EXPECT_THROW(
                SushiBLAS::RandomOps(*engine).discrete_uniform(t, -K_FLOAT32_EXACT_LIMIT - 1, 0),
                std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, IntegerValuedFillsRefuseAComplexTensor)
        {
            auto z = engine->create_tensor({4}, Core::DataType::Complex32);

            EXPECT_THROW(SushiBLAS::RandomOps(*engine).poisson(z, 1.0), std::invalid_argument);
            EXPECT_THROW(
                SushiBLAS::RandomOps(*engine).discrete_uniform(z, 0, 10),
                std::invalid_argument);
            EXPECT_THROW(SushiBLAS::RandomOps(*engine).bernoulli(z, 0.5), std::invalid_argument);
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, ToStringReadsATransposedViewThroughItsStrides)
        {
            // Row-major 2x3 holds 0..5; its transpose (i, j) reads memory i + 3j.
            auto source = create_initialized_tensor<float>({2, 3},
                                                           {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
            const Tensor view = source.transpose(0, 1);

            const std::string text = SushiBLAS::IO(*engine).to_string(view, K_PRINT_PRECISION);

            EXPECT_NE(text.find("[0.0, 3.0]"), std::string::npos) << text;
            EXPECT_NE(text.find("[1.0, 4.0]"), std::string::npos) << text;
            EXPECT_NE(text.find("[2.0, 5.0]"), std::string::npos) << text;
        }

        TEST_F(Unit_SpatialRandomIOValidationTest, SaveThenLoadReturnsTheOriginalValues)
        {
            const std::vector<float> values{1.5f, -2.25f, 3.0f, 4.75f};
            const std::string path = temp_path("round_trip.sushi");

            auto source = create_initialized_tensor<float>({2, 2}, values);
            ASSERT_EQ(SushiBLAS::IO(*engine).save(source, path), SushiBLAS::IOStatus::Ok);

            auto target = engine->create_tensor({2, 2}, Core::DataType::Float32);
            ASSERT_EQ(SushiBLAS::IO(*engine).load(target, path), SushiBLAS::IOStatus::Ok);

            EXPECT_TENSOR_NEAR(target, values, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
