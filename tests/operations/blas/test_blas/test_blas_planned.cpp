/**************************************************************************/
/* test_blas_planned.cpp                                                  */
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
 * @file test_blas_planned.cpp
 * @brief Tests binding by handle: planned and runtime tensors equal the host, unported ops refuse.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /// @brief Rows of A and C in every product of this file.
            constexpr int64_t K_ROWS = 3;

            /// @brief Columns of A and rows of B in every product of this file.
            constexpr int64_t K_DEPTH = 4;

            /// @brief Columns of B and C in every product of this file.
            constexpr int64_t K_COLUMNS = 2;

            /** @brief Returns a planned FLOAT32 tensor of the shape holding values. */
            [[nodiscard]] Tensor planned_tensor(const std::vector<int64_t>& dims,
                                                const std::vector<float>& values)
            {
                const TensorFactory factory(TestEnvironment::runtime());
                Tensor tensor = factory.planned(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                    Core::DataType::Float32);
                tensor.storage->copy_from_host(values.data(), 0, values.size() * sizeof(float));
                return tensor;
            }

            /** @brief Returns the count elements of tensor from its first, read through the host copy. */
            [[nodiscard]] std::vector<float> read_back(const Tensor& tensor, std::size_t count)
            {
                std::vector<float> values(count);
                tensor.storage->copy_to_host(
                    values.data(),
                    static_cast<std::size_t>(tensor.storage_offset) * sizeof(float),
                    count * sizeof(float));
                return values;
            }

            /** @brief Returns n small integers starting at first, so every product is exact. */
            [[nodiscard]] std::vector<float> counting(std::size_t n, float first)
            {
                std::vector<float> values(n);
                for (std::size_t i = 0; i < n; ++i)
                    values[i] = first + static_cast<float>(i % 7) - 3.0f;
                return values;
            }

            /** @brief Returns the row major product of a (rows x depth) and b (depth x columns). */
            [[nodiscard]] std::vector<float> reference_product(const std::vector<float>& a,
                                                               const std::vector<float>& b)
            {
                std::vector<float> c(static_cast<std::size_t>(K_ROWS * K_COLUMNS), 0.0f);
                for (int64_t i = 0; i < K_ROWS; ++i)
                {
                    for (int64_t j = 0; j < K_COLUMNS; ++j)
                    {
                        for (int64_t k = 0; k < K_DEPTH; ++k)
                            c[static_cast<std::size_t>(i * K_COLUMNS + j)] +=
                                a[static_cast<std::size_t>(i * K_DEPTH + k)] *
                                b[static_cast<std::size_t>(k * K_COLUMNS + j)];
                    }
                }
                return c;
            }

            /** @brief Returns the message of the std::invalid_argument fn throws, or empty. */
            template<typename Function>
            [[nodiscard]] std::string refusal_of(Function&& fn)
            {
                try
                {
                    fn();
                }
                catch (const std::invalid_argument& error)
                {
                    return error.what();
                }
                return std::string();
            }
        } // namespace

        /** @brief Records on the runtime and compares with the host. */
        class Medium_BLASPlanned : public SushiBLASTest
        {

        };

        TEST_F(Medium_BLASPlanned, GemmOverPlannedTensorsEqualsTheHostReference)
        {
            const std::vector<float> a_values = counting(K_ROWS * K_DEPTH, 1.0f);
            const std::vector<float> b_values = counting(K_DEPTH * K_COLUMNS, 2.0f);
            const Tensor a = planned_tensor({K_ROWS, K_DEPTH}, a_values);
            const Tensor b = planned_tensor({K_DEPTH, K_COLUMNS}, b_values);
            Tensor c = planned_tensor({K_ROWS, K_COLUMNS},
                                      std::vector<float>(K_ROWS * K_COLUMNS, 0.0f));

            BLASOps(recorder()).gemm(a, b, c, Core::Transpose::NoTrans, Core::Transpose::NoTrans);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(read_back(c, K_ROWS * K_COLUMNS), reference_product(a_values, b_values));
        }

        TEST_F(Medium_BLASPlanned, GemmOverPlannedViewsAtAnOffsetEqualsTheHostReference)
        {
            const std::vector<float> a_values = counting(K_ROWS * K_DEPTH, 1.0f);
            const std::vector<float> b_values = counting(K_DEPTH * K_COLUMNS, 2.0f);
            constexpr int64_t K_LEAD = 5;
            std::vector<float> whole_b(static_cast<std::size_t>(K_LEAD), -100.0f);
            whole_b.insert(whole_b.end(), b_values.begin(), b_values.end());
            const Tensor a = planned_tensor({K_ROWS, K_DEPTH}, a_values);
            const Tensor holder = planned_tensor({static_cast<int64_t>(whole_b.size())}, whole_b);
            const std::vector<int64_t> b_dims{K_DEPTH, K_COLUMNS};
            const Tensor b(holder.storage,
                           SushiRuntime::span<const int64_t>(b_dims.data(), b_dims.size()),
                           Core::DataType::Float32, K_LEAD);
            Tensor c = planned_tensor({K_ROWS, K_COLUMNS},
                                      std::vector<float>(K_ROWS * K_COLUMNS, 0.0f));

            BLASOps(recorder()).gemm(a, b, c, Core::Transpose::NoTrans, Core::Transpose::NoTrans);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(read_back(c, K_ROWS * K_COLUMNS), reference_product(a_values, b_values));
        }

        TEST_F(Medium_BLASPlanned, GemmOverRuntimeTensorsEqualsTheHostReference)
        {
            const std::vector<float> a_values = counting(K_ROWS * K_DEPTH, 1.0f);
            const std::vector<float> b_values = counting(K_DEPTH * K_COLUMNS, 2.0f);
            Tensor a = engine->create_tensor({K_ROWS, K_DEPTH});
            Tensor b = engine->create_tensor({K_DEPTH, K_COLUMNS});
            Tensor c = engine->create_tensor({K_ROWS, K_COLUMNS});
            fill_tensor(a, a_values);
            fill_tensor(b, b_values);

            BLASOps(recorder()).gemm(a, b, c, Core::Transpose::NoTrans, Core::Transpose::NoTrans);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(c, reference_product(a_values, b_values));
        }

        TEST_F(Medium_BLASPlanned, CopyOverPlannedTensorsEqualsTheHostReference)
        {
            const std::vector<float> values = counting(9, 4.0f);
            const Tensor x = planned_tensor({9}, values);
            Tensor y = planned_tensor({9}, std::vector<float>(9, 0.0f));

            BLASOps(recorder()).copy(x, y);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(read_back(y, 9), values);
        }

        TEST_F(Medium_BLASPlanned, CopyOverRuntimeTensorsEqualsTheHostReference)
        {
            const std::vector<float> values = counting(9, 4.0f);
            Tensor x = engine->create_tensor({9});
            Tensor y = engine->create_tensor({9});
            fill_tensor(x, values);

            BLASOps(recorder()).copy(x, y);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(y, values);
        }

        TEST_F(Medium_BLASPlanned, AnUnportedOperationRefusesAPlannedOperandByName)
        {
            const Tensor x = planned_tensor({9}, counting(9, 1.0f));
            Tensor y = planned_tensor({9}, counting(9, 2.0f));

            const std::string message =
                refusal_of([&] { BLASOps(recorder()).axpy(2.0, x, y); });

            EXPECT_NE(message.find("blas.lvl1.axpy"), std::string::npos) << message;
            EXPECT_NE(message.find("planned"), std::string::npos) << message;
        }

        TEST_F(Medium_BLASPlanned, AGemmWithAnEpilogueOperandRefusesAPlannedOperandByName)
        {
            const Tensor a = planned_tensor({K_ROWS, K_DEPTH}, counting(K_ROWS * K_DEPTH, 1.0f));
            const Tensor b = planned_tensor({K_DEPTH, K_COLUMNS}, counting(K_DEPTH * K_COLUMNS, 2.0f));
            Tensor c = planned_tensor({K_ROWS, K_COLUMNS},
                                      std::vector<float>(K_ROWS * K_COLUMNS, 0.0f));
            Tensor bias = engine->create_tensor({K_COLUMNS});

            const std::string message = refusal_of(
                [&]
                {
                    Dispatch::execute_gemm<Dispatch::Domain::RealAndComplex>(
                        recorder(), a, b, c, Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                        1.0, 0.0, "test.gemm_bias", "test.gemm_bias"_op,
                        Dispatch::gemm_epilogue(Kernels::GemmIdentityEpilogue{}).reads(bias));
                });

            EXPECT_NE(message.find("test.gemm_bias"), std::string::npos) << message;
            EXPECT_NE(message.find("planned"), std::string::npos) << message;
        }
    } // namespace Test
} // namespace SushiBLAS
