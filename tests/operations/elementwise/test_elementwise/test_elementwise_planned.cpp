/**************************************************************************/
/* test_elementwise_planned.cpp                                           */
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
 * @file test_elementwise_planned.cpp
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
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Elements of every tensor of this file.
            constexpr std::size_t K_COUNT = 37;

            /** @brief Returns a planned FLOAT32 vector holding values. */
            [[nodiscard]] Tensor planned_tensor(const std::vector<float>& values)
            {
                const TensorFactory factory(TestEnvironment::runtime());
                const std::vector<int64_t> dims{static_cast<int64_t>(values.size())};
                Tensor tensor = factory.planned(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                    Core::DataType::Float32);
                tensor.storage->copy_from_host(values.data(), 0, values.size() * sizeof(float));
                return tensor;
            }

            /** @brief Returns the elements of a planned vector, read through the host copy. */
            [[nodiscard]] std::vector<float> read_back(const Tensor& tensor, std::size_t count)
            {
                std::vector<float> values(count);
                tensor.storage->copy_to_host(values.data(), 0, count * sizeof(float));
                return values;
            }

            /** @brief Returns n small integers starting at first. */
            [[nodiscard]] std::vector<float> counting(std::size_t n, float first)
            {
                std::vector<float> values(n);
                for (std::size_t i = 0; i < n; ++i)
                    values[i] = first + static_cast<float>(i % 11) - 5.0f;
                return values;
            }

            /** @brief Returns what binary adds elementwise on the host. */
            [[nodiscard]] std::vector<float> reference_sum(const std::vector<float>& a,
                                                           const std::vector<float>& b)
            {
                std::vector<float> sum(a.size());
                for (std::size_t i = 0; i < a.size(); ++i)
                    sum[i] = a[i] + b[i];
                return sum;
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
        class Medium_ElementwisePlanned : public SushiBLASTest
        {

        };

        TEST_F(Medium_ElementwisePlanned, AddOverPlannedTensorsEqualsTheHostReference)
        {
            const std::vector<float> a_values = counting(K_COUNT, 1.0f);
            const std::vector<float> b_values = counting(K_COUNT, 6.0f);
            const Tensor a = planned_tensor(a_values);
            const Tensor b = planned_tensor(b_values);
            Tensor c = planned_tensor(std::vector<float>(K_COUNT, 0.0f));

            ElementwiseOps(recorder()).add(a, b, c);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(read_back(c, K_COUNT), reference_sum(a_values, b_values));
        }

        TEST_F(Medium_ElementwisePlanned, AddOverRuntimeTensorsEqualsTheHostReference)
        {
            const std::vector<float> a_values = counting(K_COUNT, 1.0f);
            const std::vector<float> b_values = counting(K_COUNT, 6.0f);
            Tensor a = engine->create_tensor({static_cast<int64_t>(K_COUNT)});
            Tensor b = engine->create_tensor({static_cast<int64_t>(K_COUNT)});
            Tensor c = engine->create_tensor({static_cast<int64_t>(K_COUNT)});
            fill_tensor(a, a_values);
            fill_tensor(b, b_values);

            ElementwiseOps(recorder()).add(a, b, c);
            ASSERT_TRUE(recording->run_succeeds());

            expect_tensor_equal(c, reference_sum(a_values, b_values));
        }

        TEST_F(Medium_ElementwisePlanned, AnInPlaceOperationOverAPlannedTensorEqualsTheHostReference)
        {
            const std::vector<float> values = counting(K_COUNT, 1.0f);
            Tensor t = planned_tensor(values);

            ElementwiseOps(recorder()).neg(t);
            ASSERT_TRUE(recording->run_succeeds());

            std::vector<float> expected = values;
            for (float& value : expected)
                value = -value;
            EXPECT_EQ(read_back(t, K_COUNT), expected);
        }

        TEST_F(Medium_ElementwisePlanned, AnUnportedOperationRefusesAPlannedOperandByName)
        {
            const Tensor src = planned_tensor(counting(K_COUNT, 1.0f));
            Tensor dst = planned_tensor(std::vector<float>(K_COUNT, 0.0f));

            const std::string message = refusal_of(
                [&] { ElementwiseOps(recorder()).convert(src, dst); });

            EXPECT_NE(message.find("convert"), std::string::npos) << message;
            EXPECT_NE(message.find("planned"), std::string::npos) << message;
        }
    } // namespace Test
} // namespace SushiBLAS
