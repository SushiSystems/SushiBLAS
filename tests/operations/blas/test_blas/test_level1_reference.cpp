/**************************************************************************/
/* test_level1_reference.cpp                                              */
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
 * @file test_level1_reference.cpp
 * @brief Tests the BLAS Level 1 reference kernels against hand-computed results.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance of every comparison in this file.
        constexpr double K_TOLERANCE = 1e-5;

        /**
         * @brief Runs every Level 1 call of a test on the reference kernels.
         */
        class Medium_Level1ReferenceTest : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and forces the reference path. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    engine->set_mode(ExecutionMode::ForceReference);
                }
        };

        TEST_F(Medium_Level1ReferenceTest, AsumSumsAbsoluteValues)
        {
            const std::vector<float> K_EXPECTED = {10.0f};

            auto x = create_initialized_tensor<float>({4}, {1.0f, -2.0f, 3.0f, -4.0f});
            auto result = engine->create_tensor({1});

            SushiBLAS::BLASOps(*engine).asum(x, result);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(result, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level1ReferenceTest, AxpyAddsScaledX)
        {
            const std::vector<float> K_EXPECTED = {12.0f, 24.0f, 36.0f};

            auto x = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto y = create_initialized_tensor<float>({3}, {10.0f, 20.0f, 30.0f});

            SushiBLAS::BLASOps(*engine).axpy(2.0, x, y);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(y, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level1ReferenceTest, CopyOverwritesEveryElement)
        {
            const std::vector<float> K_EXPECTED = {1.5f, -2.5f, 3.5f};

            auto x = create_initialized_tensor<float>({3}, {1.5f, -2.5f, 3.5f});
            auto y = create_initialized_tensor<float>({3}, {999.0f, 999.0f, 999.0f});

            SushiBLAS::BLASOps(*engine).copy(x, y);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(y, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level1ReferenceTest, DotSumsElementwiseProducts)
        {
            const std::vector<float> K_EXPECTED = {12.0f};

            auto x = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto y = create_initialized_tensor<float>({3}, {4.0f, -5.0f, 6.0f});
            auto result = engine->create_tensor({1});

            SushiBLAS::BLASOps(*engine).dot(x, y, result);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(result, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level1ReferenceTest, IamaxReturnsTheFirstIndexOfATie)
        {
            const std::vector<float> K_EXPECTED_INDEX = {1.0f};

            auto x = create_initialized_tensor<float>({4}, {1.0f, -3.0f, 3.0f, 2.0f});
            auto result = engine->create_tensor({1}, Core::DataType::Int64);

            SushiBLAS::BLASOps(*engine).iamax(x, result);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(result, K_EXPECTED_INDEX, 0.0);
        }

        TEST_F(Medium_Level1ReferenceTest, IamaxFindsANegativeMaximumAtTheEnd)
        {
            const std::vector<float> K_EXPECTED_INDEX = {3.0f};

            auto x = create_initialized_tensor<float>({4}, {0.5f, -1.0f, 2.0f, -7.0f});
            auto result = engine->create_tensor({1}, Core::DataType::Int64);

            SushiBLAS::BLASOps(*engine).iamax(x, result);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(result, K_EXPECTED_INDEX, 0.0);
        }

        TEST_F(Medium_Level1ReferenceTest, Nrm2ReturnsTheEuclideanNorm)
        {
            const std::vector<float> K_EXPECTED = {7.0f};

            auto x = create_initialized_tensor<float>({3}, {2.0f, -3.0f, 6.0f});
            auto result = engine->create_tensor({1});

            SushiBLAS::BLASOps(*engine).nrm2(x, result);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(result, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level1ReferenceTest, RotAppliesThePlaneRotation)
        {
            const std::vector<float> K_EXPECTED_X = {3.0f, 4.4f};
            const std::vector<float> K_EXPECTED_Y = {1.0f, 0.8f};

            auto x = create_initialized_tensor<float>({2}, {1.0f, 2.0f});
            auto y = create_initialized_tensor<float>({2}, {3.0f, 4.0f});

            SushiBLAS::BLASOps(*engine).rot(x, y, 0.6, 0.8);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED_X, K_TOLERANCE);
            EXPECT_TENSOR_NEAR(y, K_EXPECTED_Y, K_TOLERANCE);
        }

        TEST_F(Medium_Level1ReferenceTest, ScalMultipliesEveryElement)
        {
            const std::vector<float> K_EXPECTED = {-1.0f, 2.0f, -3.0f};

            auto x = create_initialized_tensor<float>({3}, {2.0f, -4.0f, 6.0f});

            SushiBLAS::BLASOps(*engine).scal(-0.5, x);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED, K_TOLERANCE);
        }

        TEST_F(Medium_Level1ReferenceTest, SwapExchangesBothVectors)
        {
            const std::vector<float> K_EXPECTED_X = {7.0f, 8.0f, 9.0f};
            const std::vector<float> K_EXPECTED_Y = {1.0f, 2.0f, 3.0f};

            auto x = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto y = create_initialized_tensor<float>({3}, {7.0f, 8.0f, 9.0f});

            SushiBLAS::BLASOps(*engine).swap(x, y);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(x, K_EXPECTED_X, K_TOLERANCE);
            EXPECT_TENSOR_NEAR(y, K_EXPECTED_Y, K_TOLERANCE);
        }
    } // namespace Test
} // namespace SushiBLAS
