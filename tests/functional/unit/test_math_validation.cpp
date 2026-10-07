/**************************************************************************/
/* test_math_validation.cpp                                               */
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
 * @file test_math_validation.cpp
 * @brief Tests the operand checks and numeric edge cases of the elementwise, nonlinear and reduction ops.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Input whose softplus must not overflow. */
            constexpr float K_SOFTPLUS_LARGE_INPUT = 100.0f;

            /** @brief softplus(100) = 100 + log1p(exp(-100)), and exp(-100) is about 3.7e-44. */
            constexpr float K_SOFTPLUS_LARGE_EXPECTED = 100.0f;

            /** @brief Absolute tolerance on softplus(100). */
            constexpr float K_SOFTPLUS_TOLERANCE = 1e-4f;

            /** @brief Input whose softplus is exp(-100), about 3.7e-44. */
            constexpr float K_SOFTPLUS_SMALL_INPUT = -100.0f;

            /** @brief Upper bound on softplus(-100); exp(-100) lies far below it. */
            constexpr float K_SOFTPLUS_SMALL_BOUND = 1e-30f;
        }

        class Unit_MathValidationTest : public SushiBLASTest
        {
            protected:
                /** @brief Runs the recorded graph, then returns the first FLOAT32 of t. */
                float run_and_read(const Tensor& t)
                {
                    engine->execute().wait();
                    float host = 0.0f;
                    TestEnvironment::get_context().get_queue()
                        .memcpy(&host, t.data(), sizeof(float)).wait();
                    return host;
                }

                /** @brief Returns the FLOAT32 sum of data, computed by ReductionOps::sum. */
                float sum_of(const std::vector<float>& data)
                {
                    auto t = create_initialized_tensor<float>({static_cast<int64_t>(data.size())}, data);
                    auto result = engine->create_tensor({1}, Core::DataType::FLOAT32);
                    engine->reductions().sum(t, result);
                    return run_and_read(result);
                }
        };

        TEST_F(Unit_MathValidationTest, AddRejectsMismatchedShapes)
        {
            auto a = engine->create_tensor({2, 3}, Core::DataType::FLOAT32);
            auto b = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);
            auto c = engine->create_tensor({2, 3}, Core::DataType::FLOAT32);
            EXPECT_THROW(engine->elementwise().add(a, b, c), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, AddRejectsTransposedView)
        {
            auto a = engine->create_tensor({2, 3}, Core::DataType::FLOAT32).transpose(0, 1);
            auto b = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);
            auto c = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);
            EXPECT_THROW(engine->elementwise().add(a, b, c), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, AddRejectsTensorWithoutStorage)
        {
            Tensor a(SushiRuntime::sushi_ptr<Storage>(), {4});
            auto b = engine->create_tensor({4}, Core::DataType::FLOAT32);
            auto c = engine->create_tensor({4}, Core::DataType::FLOAT32);
            EXPECT_THROW(engine->elementwise().add(a, b, c), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, UnaryInPlaceRejectsTensorWithoutStorage)
        {
            Tensor t(SushiRuntime::sushi_ptr<Storage>(), {4});
            EXPECT_THROW(engine->elementwise().exp(t), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, SumRejectsTensorWithoutStorage)
        {
            Tensor t(SushiRuntime::sushi_ptr<Storage>(), {4});
            auto result = engine->create_tensor({1}, Core::DataType::FLOAT32);
            EXPECT_THROW(engine->reductions().sum(t, result), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, VarRejectsEmptyTensor)
        {
            Tensor t(SushiRuntime::sushi_ptr<Storage>(), {0});
            auto result = engine->create_tensor({1}, Core::DataType::FLOAT32);
            EXPECT_THROW(engine->reductions().var(t, result), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, SoftplusOfLargeInputIsFinite)
        {
            auto t = create_initialized_tensor<float>({1}, {K_SOFTPLUS_LARGE_INPUT});
            engine->nonlinear().softplus(t);
            EXPECT_NEAR(run_and_read(t), K_SOFTPLUS_LARGE_EXPECTED, K_SOFTPLUS_TOLERANCE);
        }

        TEST_F(Unit_MathValidationTest, SoftplusOfLargeNegativeInputIsTinyAndNonNegative)
        {
            auto t = create_initialized_tensor<float>({1}, {K_SOFTPLUS_SMALL_INPUT});
            engine->nonlinear().softplus(t);
            const float value = run_and_read(t);
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_GE(value, 0.0f);
            EXPECT_LE(value, K_SOFTPLUS_SMALL_BOUND);
        }

        TEST_F(Unit_MathValidationTest, ReluRejectsComplexTensor)
        {
            auto t = engine->create_tensor({2}, Core::DataType::COMPLEX32);
            EXPECT_THROW(engine->nonlinear().relu(t), std::runtime_error);
        }

        TEST_F(Unit_MathValidationTest, SumWithPositiveInfinityIsPositiveInfinity)
        {
            const float inf = std::numeric_limits<float>::infinity();
            const float value = sum_of({inf, 1.0f, 2.0f});
            EXPECT_TRUE(std::isinf(value));
            EXPECT_GT(value, 0.0f);
        }

        TEST_F(Unit_MathValidationTest, SumOfOppositeInfinitiesIsNaN)
        {
            const float inf = std::numeric_limits<float>::infinity();
            EXPECT_TRUE(std::isnan(sum_of({inf, -inf})));
        }

    } // namespace Test
} // namespace SushiBLAS
