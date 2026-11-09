/**************************************************************************/
/* test_operand_refusals.cpp                                              */
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
 * @file test_operand_refusals.cpp
 * @brief Tests, on a host engine, which operands the elementwise, nonlinear and reduction ops refuse at record time.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
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

            /** @brief Names one operation and the call that records it on a tensor without storage. */
            struct StorageLessRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief Records the operation over operands whose first tensor has no storage. */
                std::function<void(TaskRecorder&, Engine&)> record;
            };

            /** @brief Returns a FLOAT32 tensor of four elements with no storage. */
            Tensor detached_four()
            {
                return Tensor(SushiRuntime::sushi_ptr<Storage>(), {4});
            }

            /** @brief Names a row by its name. */
            std::string storage_less_name(const ::testing::TestParamInfo<StorageLessRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /// @brief Records operations on a host engine; the refusals are decided on shape, stride, dtype and storage.
        class Small_OperandRefusal : public HostEngineTest {};

        /// @brief Tests that add refuses operands whose shapes differ.
        TEST_F(Small_OperandRefusal, AddRejectsMismatchedShapes)
        {
            auto a = engine->create_tensor({2, 3}, Core::DataType::Float32);
            auto b = engine->create_tensor({3, 2}, Core::DataType::Float32);
            auto c = engine->create_tensor({2, 3}, Core::DataType::Float32);

            EXPECT_THROW(ElementwiseOps(recorder()).add(a, b, c), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        /// @brief Tests that add refuses a transposed view, whose strides are not dense.
        TEST_F(Small_OperandRefusal, AddRejectsTransposedView)
        {
            auto a = engine->create_tensor({2, 3}, Core::DataType::Float32).transpose(0, 1);
            auto b = engine->create_tensor({3, 2}, Core::DataType::Float32);
            auto c = engine->create_tensor({3, 2}, Core::DataType::Float32);

            EXPECT_THROW(ElementwiseOps(recorder()).add(a, b, c), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        /// @brief Tests that a real-only nonlinear op refuses a complex tensor.
        TEST_F(Small_OperandRefusal, ReLURejectsComplexTensor)
        {
            auto t = engine->create_tensor({2}, Core::DataType::Complex32);

            EXPECT_THROW(NonLinearOps(recorder()).relu(t), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        /// @brief Names one real-only nonlinear op and the call that records it on a tensor.
        struct NonLinearRow
        {
            /** @brief The row name, which gtest takes as the suffix of the test. */
            std::string name;

            /** @brief Records the operation in place on t. */
            std::function<void(NonLinearOps&, Tensor&)> record;
        };

        /// @brief Records a real-only nonlinear op over a complex tensor, one op per row.
        class Small_ComplexNonLinear : public Small_OperandRefusal,
                                       public ::testing::WithParamInterface<NonLinearRow> {};

        /// @brief Tests that a real-only nonlinear op refuses a complex tensor and records nothing.
        TEST_P(Small_ComplexNonLinear, IsRefusedAndRecordsNothing)
        {
            auto t = engine->create_tensor({2}, Core::DataType::Complex32);
            NonLinearOps ops(recorder());

            EXPECT_THROW(GetParam().record(ops, t), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        INSTANTIATE_TEST_SUITE_P(
            Operations, Small_ComplexNonLinear,
            ::testing::Values(
                NonLinearRow{"GELU", [](NonLinearOps& o, Tensor& t) { o.gelu(t); }},
                NonLinearRow{"SiLU", [](NonLinearOps& o, Tensor& t) { o.silu(t); }},
                NonLinearRow{"Softplus", [](NonLinearOps& o, Tensor& t) { o.softplus(t); }},
                NonLinearRow{"LeakyReLU", [](NonLinearOps& o, Tensor& t) { o.leaky_relu(t); }}),
            [](const ::testing::TestParamInfo<NonLinearRow>& info) { return info.param.name; });

        /// @brief Tests that a dtype mismatch names the operand and both dtypes.
        TEST_F(Small_OperandRefusal, AMismatchedDtypeNamesTheOperand)
        {
            auto a = engine->create_tensor({4}, Core::DataType::Float32);
            auto b = engine->create_tensor({4}, Core::DataType::Float64);
            auto c = engine->create_tensor({4}, Core::DataType::Float32);

            const std::string text = refusal_of([&] { ElementwiseOps(recorder()).add(a, b, c); });

            EXPECT_TRUE(holds(text, "operand 'B'")) << text;
            EXPECT_TRUE(holds(text, "FLOAT64")) << text;
            EXPECT_TRUE(holds(text, "FLOAT32")) << text;
        }

        /// @brief Runs one record call per row over a first operand that has no storage.
        class Small_StorageLessOperand : public Small_OperandRefusal,
                                         public ::testing::WithParamInterface<StorageLessRow> {};

        /// @brief Tests that an operation refuses a tensor without storage and records nothing.
        TEST_P(Small_StorageLessOperand, IsRefusedAndRecordsNothing)
        {
            EXPECT_THROW(GetParam().record(recorder(), *engine), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        INSTANTIATE_TEST_SUITE_P(
            Operations, Small_StorageLessOperand,
            ::testing::Values(
                StorageLessRow{"Add",
                               [](TaskRecorder& r, Engine& e)
                               {
                                   Tensor a = detached_four();
                                   Tensor b = e.create_tensor({4}, Core::DataType::Float32);
                                   Tensor c = e.create_tensor({4}, Core::DataType::Float32);
                                   ElementwiseOps(r).add(a, b, c);
                               }},
                StorageLessRow{"UnaryInPlace",
                               [](TaskRecorder& r, Engine&)
                               {
                                   Tensor t = detached_four();
                                   ElementwiseOps(r).exp(t);
                               }},
                StorageLessRow{"Sum",
                               [](TaskRecorder& r, Engine& e)
                               {
                                   Tensor t = detached_four();
                                   Tensor result = e.create_tensor({1}, Core::DataType::Float32);
                                   ReductionOps(r).sum(t, result);
                               }}),
            storage_less_name);
    } // namespace Test
} // namespace SushiBLAS
