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
 * @file test_math_validation.cpp
 * @brief Tests the device-only operand refusal, the mode independence of operand errors and the softplus range.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <exception>
#include <functional>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

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

            /** @brief Absolute tolerance: 128 float roundings of a value of order one. */
            constexpr float K_SOFTPLUS_TOLERANCE =
                static_cast<float>(absolute_tolerance<float>(128));

            /** @brief Input whose softplus is exp(-100), about 3.7e-44. */
            constexpr float K_SOFTPLUS_SMALL_INPUT = -100.0f;

            /** @brief Upper bound: the smallest normal float, so exp(-100) must come back subnormal or zero. */
            constexpr float K_SOFTPLUS_SMALL_BOUND = std::numeric_limits<float>::min();

            /** @brief Holds the operands the family rows share. */
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

                /** @brief INT64 output of one element, for argmax. */
                Tensor index;
            };

            /** @brief Names one operation family and the call that records one of its operations. */
            struct FamilyRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief Records the operation over the shared operands. */
                std::function<void(TaskRecorder&, RecordOperands&, RNGStream&)> record;
            };

            /** @brief Names one execution mode. */
            struct ModeRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The mode the row records under. */
                ExecutionMode mode;
            };

            /** @brief Returns one record call per seam and per hand-recorded source. */
            std::vector<FamilyRow> family_rows()
            {
                return {
                    {"Add", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { ElementwiseOps(r).add(on.a, on.b, on.c); }},
                    {"Clamp", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { ElementwiseOps(r).clamp(on.c, 0.0, 1.0); }},
                    {"Convert", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { ElementwiseOps(r).convert(on.a, on.c); }},
                    {"ReLU", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { NonLinearOps(r).relu(on.c); }},
                    {"EluBackward", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { NonLinearOps(r).elu_backward(on.a, on.b, on.c, 1.0); }},
                    {"Equal", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { LogicOps(r).equal(on.a, on.b, on.c); }},
                    {"LogicalNot", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { LogicOps(r).logical_not(on.a, on.c); }},
                    {"Where", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { LogicOps(r).where(on.a, on.a, on.b, on.c); }},
                    {"All", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { LogicOps(r).all(on.a, on.scalar); }},
                    {"Uniform", [](TaskRecorder& r, RecordOperands& on, RNGStream& rng)
                     { RandomOps(r, rng).uniform(on.c); }},
                    {"Constant", [](TaskRecorder& r, RecordOperands& on, RNGStream& rng)
                     { RandomOps(r, rng).constant(on.c, 1.0); }},
                    {"Shuffle", [](TaskRecorder& r, RecordOperands& on, RNGStream& rng)
                     { RandomOps(r, rng).shuffle(on.c); }},
                    {"Orthogonal", [](TaskRecorder& r, RecordOperands& on, RNGStream& rng)
                     { RandomOps(r, rng).orthogonal(on.matrix); }},
                    {"Sum", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { ReductionOps(r).sum(on.a, on.scalar); }},
                    {"Argmax", [](TaskRecorder& r, RecordOperands& on, RNGStream&)
                     { ReductionOps(r).argmax(on.a, on.index); }}};
            }

            /** @brief Names a family row by its name. */
            std::string family_row_name(const ::testing::TestParamInfo<FamilyRow>& info)
            {
                return info.param.name;
            }

            /** @brief Names a mode row by its name. */
            std::string mode_row_name(const ::testing::TestParamInfo<ModeRow>& info)
            {
                return info.param.name;
            }

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
        } // namespace

        class Medium_MathValidationTest : public SushiBLASTest
        {
            protected:
                /** @brief Runs the recorded graph, then returns the first FLOAT32 of t. */
                float run_and_read(const Tensor& t)
                {
                    recording->run().value();
                    float host = 0.0f;
                    TestEnvironment::queue()
                        .memcpy(&host, t.data(), sizeof(float)).wait();
                    return host;
                }

                /** @brief Returns the operands the family rows share, in one residency. */
                RecordOperands operands_in(SushiRuntime::Residency residency)
                {
                    const auto make = [&](std::initializer_list<int64_t> dims,
                                          Core::DataType dtype)
                    {
                        return engine->create_tensor(dims, dtype, residency);
                    };

                    return RecordOperands{make({4}, Core::DataType::Float32),
                                          make({4}, Core::DataType::Float32),
                                          make({4}, Core::DataType::Float32),
                                          make({1}, Core::DataType::Float32),
                                          make({2, 2}, Core::DataType::Float32),
                                          make({1}, Core::DataType::Int64)};
                }

                /**
                 * @brief Returns the same operands in device-only USM, or nothing without such USM.
                 * @return Empty when an allocation fails or a pointer is not usm::alloc::device.
                 */
                std::optional<RecordOperands> device_only_operands()
                {
                    try
                    {
                        RecordOperands on = operands_in(SushiRuntime::Residency::Device);
                        const sycl::context context =
                            TestEnvironment::queue().get_context();

                        for (const Tensor* t :
                             {&on.a, &on.b, &on.c, &on.scalar, &on.matrix, &on.index})
                        {
                            if (sycl::get_pointer_type(t->data(), context) !=
                                sycl::usm::alloc::device)
                                return std::nullopt;
                        }
                        return on;
                    }
                    catch (const std::exception&)
                    {
                        return std::nullopt;
                    }
                }
        };

        TEST_F(Medium_MathValidationTest, SoftplusOfLargeInputIsFinite)
        {
            auto t = create_initialized_tensor<float>({1}, {K_SOFTPLUS_LARGE_INPUT});
            SushiBLAS::NonLinearOps(recorder()).softplus(t);
            EXPECT_NEAR(run_and_read(t), K_SOFTPLUS_LARGE_EXPECTED, K_SOFTPLUS_TOLERANCE);
        }

        TEST_F(Medium_MathValidationTest, SoftplusOfLargeNegativeInputIsTinyAndNonNegative)
        {
            auto t = create_initialized_tensor<float>({1}, {K_SOFTPLUS_SMALL_INPUT});
            SushiBLAS::NonLinearOps(recorder()).softplus(t);
            const float value = run_and_read(t);
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_GE(value, 0.0f);
            EXPECT_LE(value, K_SOFTPLUS_SMALL_BOUND);
        }

        /// @brief Records one operation of each family per row, over operands in one residency.
        class Medium_FamilyRecord : public Medium_MathValidationTest,
                                    public ::testing::WithParamInterface<FamilyRow> {};

        /**
         * @brief Tests that ForceReference records an operation of every family over shared operands.
         * @see tests/library/umbrella/README.md
         */
        TEST_P(Medium_FamilyRecord, IsRecordedUnderForceReference)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            RecordOperands shared = operands_in(SushiRuntime::Residency::Shared);

            EXPECT_NO_THROW(GetParam().record(probe.recorder(), shared, engine->rng()));
            EXPECT_NE(0u, probe.take_recorded_count());
        }

        INSTANTIATE_TEST_SUITE_P(Families, Medium_FamilyRecord,
                                 ::testing::ValuesIn(family_rows()), family_row_name);

        /// @brief Holds the operands in device-only USM, and skips each test when the device offers none.
        class Medium_DeviceOnlyOperand : public Medium_MathValidationTest,
                                         public ::testing::WithParamInterface<FamilyRow>
        {
            protected:
                /** @brief Allocates the operands in device-only USM or skips the test. */
                void SetUp() override
                {
                    Medium_MathValidationTest::SetUp();
                    device_only = device_only_operands();
                    if (!device_only)
                        GTEST_SKIP() << "This device offers no device-only USM.";
                }

                /// @brief The operands, in device-only USM.
                std::optional<RecordOperands> device_only;
        };

        /**
         * @brief Tests that ForceReference refuses a device-only operand in every family.
         * @see tests/library/umbrella/README.md
         */
        TEST_P(Medium_DeviceOnlyOperand, IsRefusedUnderForceReference)
        {
            GraphProbe probe(ExecutionMode::ForceReference);

            const std::string text = refusal_of(
                [&] { GetParam().record(probe.recorder(), *device_only, engine->rng()); });

            EXPECT_TRUE(holds(text, "ForceReference")) << text;
            EXPECT_EQ(0u, probe.take_recorded_count());
        }

        INSTANTIATE_TEST_SUITE_P(Families, Medium_DeviceOnlyOperand,
                                 ::testing::ValuesIn(family_rows()), family_row_name);

        /// @brief Records the same bad call under one mode per row.
        class Medium_OperandErrorMode : public Medium_MathValidationTest,
                                        public ::testing::WithParamInterface<ModeRow>
        {
            protected:
                /** @brief Returns the text add refuses mismatched shapes with under mode. */
                std::string add_refusal_under(ExecutionMode mode)
                {
                    auto a = engine->create_tensor({2, 3});
                    auto b = engine->create_tensor({3, 2});
                    auto c = engine->create_tensor({2, 3});

                    GraphProbe probe(mode);
                    return refusal_of([&] { SushiBLAS::ElementwiseOps(probe.recorder()).add(a, b, c); });
                }
        };

        /**
         * @brief Tests that an operand error reads the same under every mode.
         * @see tests/library/umbrella/README.md
         */
        TEST_P(Medium_OperandErrorMode, ReadsTheSameAsUnderAuto)
        {
            const std::string text = add_refusal_under(GetParam().mode);

            EXPECT_TRUE(holds(text, "'B'")) << text;
            EXPECT_FALSE(holds(text, "ForceReference")) << text;
            EXPECT_EQ(add_refusal_under(ExecutionMode::Auto), text);
        }

        /**
         * @brief Tests that an add over empty operands records nothing under every mode.
         * @see tests/library/umbrella/README.md
         */
        TEST_P(Medium_OperandErrorMode, AnEmptyCallRecordsNothing)
        {
            auto e0 = engine->create_tensor({0});
            auto e1 = engine->create_tensor({0});
            auto e2 = engine->create_tensor({0});

            GraphProbe probe(GetParam().mode);

            EXPECT_FALSE(SushiBLAS::ElementwiseOps(probe.recorder()).add(e0, e1, e2).is_valid());
            EXPECT_EQ(0u, probe.take_recorded_count());
        }

        INSTANTIATE_TEST_SUITE_P(
            Modes, Medium_OperandErrorMode,
            ::testing::Values(ModeRow{"Auto", ExecutionMode::Auto},
                              ModeRow{"ForceSYCL", ExecutionMode::ForceSYCL},
                              ModeRow{"ForceReference", ExecutionMode::ForceReference}),
            mode_row_name);
    } // namespace Test
} // namespace SushiBLAS
