/**************************************************************************/
/* test_scalar_operand.cpp                                                */
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
 * @file test_scalar_operand.cpp
 * @brief Tests the record-time checks on a one-element scalar operand.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas/scalar_operand.hpp>
#include "structured_operand_test.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;

            /** @brief One tensor a scalar operand check meets, built on demand. */
            struct TensorRow
            {
                const char* name;              /**< Names the row. */
                int32_t rank;                  /**< The rank the tensor has. */
                std::function<Tensor()> build; /**< Builds the tensor. */
            };

            /** @brief Returns the tensors of one stored element, whatever their rank. */
            std::vector<TensorRow> scalar_rows()
            {
                return {
                    {"RankZero", 0, [] { return host_matrix(std::initializer_list<int64_t>{}); }},
                    {"RankOne", 1, [] { return host_matrix({1}); }},
                    {"RankTwo", 2, [] { return host_matrix({1, 1}); }},
                };
            }

            /** @brief Returns the tensors of another element count than one. */
            std::vector<TensorRow> wrong_count_rows()
            {
                return {
                    {"TwoElements", 1, [] { return host_matrix({2}); }},
                    {"NoElement", 1, [] { return detached_matrix({0}); }},
                };
            }

            /** @brief Names the suite of the tensors that are one stored element. */
            class Small_ScalarOperandAccepts : public ::testing::TestWithParam<TensorRow>
            {
            };

            /** @brief Names the suite of the tensors of another element count. */
            class Small_ScalarOperandCount : public ::testing::TestWithParam<TensorRow>
            {
            };

            /** @brief One dtype the call expects while the tensor holds FLOAT32. */
            struct DtypeRow
            {
                const char* name; /**< Names the row. */
                DataType expected; /**< The dtype the call expects. */
            };

            /** @brief Names the suite of the dtype rows. */
            class Small_ScalarOperandDtype : public ::testing::TestWithParam<DtypeRow>
            {
            };
        } // namespace

        TEST_P(Small_ScalarOperandAccepts, OneElementOfAnyRankIsAccepted)
        {
            const TensorRow row = GetParam();
            const Tensor t = row.build();
            ASSERT_EQ(row.rank, t.rank);
            ASSERT_EQ(1, t.num_elements);

            EXPECT_NO_THROW(Dispatch::require_scalar_operand(t, DataType::Float32, "op", "alpha"));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ScalarOperandAccepts, ::testing::ValuesIn(scalar_rows()),
            [](const ::testing::TestParamInfo<TensorRow>& info) { return std::string(info.param.name); });

        TEST_P(Small_ScalarOperandCount, AnotherElementCountIsRefused)
        {
            const Tensor t = GetParam().build();

            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ScalarOperandCount, ::testing::ValuesIn(wrong_count_rows()),
            [](const ::testing::TestParamInfo<TensorRow>& info) { return std::string(info.param.name); });

        TEST(Small_ScalarOperand, AMissingStorageIsRefused)
        {
            const Tensor t = detached_matrix({1});

            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);
        }

        TEST_P(Small_ScalarOperandDtype, AnotherDtypeIsRefused)
        {
            const Tensor t = host_matrix({1});

            EXPECT_THROW(Dispatch::require_scalar_operand(t, GetParam().expected, "op", "alpha"),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ScalarOperandDtype,
            ::testing::Values(DtypeRow{"Float64", DataType::Float64}, DtypeRow{"Int32", DataType::Int32}),
            [](const ::testing::TestParamInfo<DtypeRow>& info) { return std::string(info.param.name); });
    } // namespace Test
} // namespace SushiBLAS
