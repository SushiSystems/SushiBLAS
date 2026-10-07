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
#include <string>
#include <cstdint>
#include <stdexcept>
#include <functional>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"
#include "../../common/graph_probe.hpp"

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

            /** @brief Names one record call of a family that has no reference kernel. */
            struct RecordCase
            {
                /** @brief The operation name, for the failure message. */
                const char* name;

                /** @brief Records the operation. */
                std::function<void()> record;
            };

            /** @brief Holds the operands the record cases share. */
            struct RecordOperands
            {
                /** @brief First input of four elements. */
                Tensor a;

                /** @brief Second input of four elements. */
                Tensor b;

                /** @brief Output of four elements. */
                Tensor c;

                /** @brief Output of one element. */
                Tensor scalar;

                /** @brief Two by two matrix for orthogonal. */
                Tensor matrix;
            };

            /** @brief Returns what record throws as std::invalid_argument, or an empty string. */
            std::string refusal_of(const std::function<void()>& record)
            {
                try
                {
                    record();
                }
                catch (const std::invalid_argument& error)
                {
                    return error.what();
                }
                return std::string();
            }

            /** @brief Reports whether text holds part. */
            bool holds(const std::string& text, const char* part)
            {
                return text.find(part) != std::string::npos;
            }
        }

        class Unit_MathValidationTest : public SushiBLASTest
        {
            protected:
                /** @brief Runs the recorded graph, then returns the first FLOAT32 of t. */
                float run_and_read(const Tensor& t)
                {
                    engine->execute();
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
                    SushiBLAS::ReductionOps(*engine).sum(t, result);
                    return run_and_read(result);
                }

                /** @brief Returns one record call per seam and per hand-recorded source. */
                std::vector<RecordCase> record_cases(TaskRecorder& recorder, RecordOperands& on)
                {
                    RngStream& rng = engine->get_rng();
                    return {
                        {"add", [&] { SushiBLAS::ElementwiseOps(recorder).add(on.a, on.b, on.c); }},
                        {"clamp", [&] { SushiBLAS::ElementwiseOps(recorder).clamp(on.c, 0.0, 1.0); }},
                        {"convert", [&] { SushiBLAS::ElementwiseOps(recorder).convert(on.a, on.c); }},
                        {"relu", [&] { SushiBLAS::NonLinearOps(recorder).relu(on.c); }},
                        {"elu_backward",
                         [&] { SushiBLAS::NonLinearOps(recorder).elu_backward(on.a, on.b, on.c, 1.0); }},
                        {"equal", [&] { SushiBLAS::LogicOps(recorder).equal(on.a, on.b, on.c); }},
                        {"logical_not", [&] { SushiBLAS::LogicOps(recorder).logical_not(on.a, on.c); }},
                        {"where", [&] { SushiBLAS::LogicOps(recorder).where(on.a, on.a, on.b, on.c); }},
                        {"all", [&] { SushiBLAS::LogicOps(recorder).all(on.a, on.scalar); }},
                        {"uniform", [&] { SushiBLAS::RandomOps(recorder, rng).uniform(on.c); }},
                        {"constant", [&] { SushiBLAS::RandomOps(recorder, rng).constant(on.c, 1.0); }},
                        {"shuffle", [&] { SushiBLAS::RandomOps(recorder, rng).shuffle(on.c); }},
                        {"orthogonal", [&] { SushiBLAS::RandomOps(recorder, rng).orthogonal(on.matrix); }},
                    };
                }
        };

        TEST_F(Unit_MathValidationTest, AddRejectsMismatchedShapes)
        {
            auto a = engine->create_tensor({2, 3}, Core::DataType::FLOAT32);
            auto b = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);
            auto c = engine->create_tensor({2, 3}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).add(a, b, c), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, AddRejectsTransposedView)
        {
            auto a = engine->create_tensor({2, 3}, Core::DataType::FLOAT32).transpose(0, 1);
            auto b = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);
            auto c = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).add(a, b, c), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, AddRejectsTensorWithoutStorage)
        {
            Tensor a(SushiRuntime::sushi_ptr<Storage>(), {4});
            auto b = engine->create_tensor({4}, Core::DataType::FLOAT32);
            auto c = engine->create_tensor({4}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).add(a, b, c), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, UnaryInPlaceRejectsTensorWithoutStorage)
        {
            Tensor t(SushiRuntime::sushi_ptr<Storage>(), {4});
            EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).exp(t), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, SumRejectsTensorWithoutStorage)
        {
            Tensor t(SushiRuntime::sushi_ptr<Storage>(), {4});
            auto result = engine->create_tensor({1}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::ReductionOps(*engine).sum(t, result), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, VarRejectsEmptyTensor)
        {
            Tensor t(SushiRuntime::sushi_ptr<Storage>(), {0});
            auto result = engine->create_tensor({1}, Core::DataType::FLOAT32);
            EXPECT_THROW(SushiBLAS::ReductionOps(*engine).var(t, result), std::invalid_argument);
        }

        TEST_F(Unit_MathValidationTest, SoftplusOfLargeInputIsFinite)
        {
            auto t = create_initialized_tensor<float>({1}, {K_SOFTPLUS_LARGE_INPUT});
            SushiBLAS::NonLinearOps(*engine).softplus(t);
            EXPECT_NEAR(run_and_read(t), K_SOFTPLUS_LARGE_EXPECTED, K_SOFTPLUS_TOLERANCE);
        }

        TEST_F(Unit_MathValidationTest, SoftplusOfLargeNegativeInputIsTinyAndNonNegative)
        {
            auto t = create_initialized_tensor<float>({1}, {K_SOFTPLUS_SMALL_INPUT});
            SushiBLAS::NonLinearOps(*engine).softplus(t);
            const float value = run_and_read(t);
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_GE(value, 0.0f);
            EXPECT_LE(value, K_SOFTPLUS_SMALL_BOUND);
        }

        TEST_F(Unit_MathValidationTest, ReluRejectsComplexTensor)
        {
            auto t = engine->create_tensor({2}, Core::DataType::COMPLEX32);
            EXPECT_THROW(SushiBLAS::NonLinearOps(*engine).relu(t), std::invalid_argument);
        }

        /**
         * @brief Tests that FORCE_REFERENCE refuses each family before it reads an operand.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_MathValidationTest, ForceReferenceRefusesEveryFamilyWithoutAReferenceKernel)
        {
            // A tensor without storage fails the first operand check, so the mode is named first.
            const SushiRuntime::sushi_ptr<Storage> none;
            RecordOperands bare{Tensor(none, {4}), Tensor(none, {4}), Tensor(none, {4}),
                                Tensor(none, {1}), Tensor(none, {2, 2})};

            GraphProbe reference(ExecutionMode::FORCE_REFERENCE);
            for (const RecordCase& c : record_cases(reference.get_recorder(), bare))
            {
                EXPECT_TRUE(holds(refusal_of(c.record), "FORCE_REFERENCE"))
                    << c.name << " did not refuse the mode first";
            }
            EXPECT_TRUE(reference.take_nodes().empty());

            RecordOperands stored{engine->create_tensor({4}), engine->create_tensor({4}),
                                  engine->create_tensor({4}), engine->create_tensor({1}),
                                  engine->create_tensor({2, 2})};

            GraphProbe sycl_only(ExecutionMode::FORCE_SYCL);
            for (const RecordCase& c : record_cases(sycl_only.get_recorder(), stored))
            {
                EXPECT_NO_THROW(c.record()) << c.name;
                EXPECT_FALSE(sycl_only.take_nodes().empty()) << c.name << " recorded nothing";
            }
        }

        /**
         * @brief Tests that FORCE_REFERENCE is reported for an empty tensor and before an operand error.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_MathValidationTest, ForceReferenceRefusesAnEmptyTensorToo)
        {
            GraphProbe probe(ExecutionMode::FORCE_REFERENCE);
            SushiBLAS::ElementwiseOps ops(probe.get_recorder());

            auto e0 = engine->create_tensor({0});
            auto e1 = engine->create_tensor({0});
            auto e2 = engine->create_tensor({0});
            EXPECT_THROW(ops.add(e0, e1, e2), std::invalid_argument);

            auto a = engine->create_tensor({2, 3});
            auto b = engine->create_tensor({3, 2});
            auto c = engine->create_tensor({2, 3});
            const std::string text = refusal_of([&] { ops.add(a, b, c); });
            EXPECT_TRUE(holds(text, "FORCE_REFERENCE")) << text;
            EXPECT_FALSE(holds(text, "operand '")) << text;
            EXPECT_TRUE(probe.take_nodes().empty());
        }

        /**
         * @brief Tests that a dtype mismatch names the operand and both dtypes.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_MathValidationTest, AMismatchedDtypeNamesTheOperand)
        {
            auto a = engine->create_tensor({4}, Core::DataType::FLOAT32);
            auto b = engine->create_tensor({4}, Core::DataType::FLOAT64);
            auto c = engine->create_tensor({4}, Core::DataType::FLOAT32);

            const std::string text =
                refusal_of([&] { SushiBLAS::ElementwiseOps(*engine).add(a, b, c); });
            EXPECT_TRUE(holds(text, "operand 'B'")) << text;
            EXPECT_TRUE(holds(text, "FLOAT64")) << text;
            EXPECT_TRUE(holds(text, "FLOAT32")) << text;
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
