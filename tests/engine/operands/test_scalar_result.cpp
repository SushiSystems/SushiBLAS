/**************************************************************************/
/* test_scalar_result.cpp                                                 */
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
 * @file test_scalar_result.cpp
 * @brief Tests the real dtype of a scalar result and the check on the tensor that receives it.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/operands/scalar_result.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds result tensors on the fixture's engine. */
        class Medium_ScalarResult : public SushiBLASTest
        {

        };

        namespace
        {
            /** @brief Names a dtype and the real dtype of its parts. */
            struct RealDtypeRow
            {
                const char* name;         /**< The row name a failure shows. */
                Core::DataType dtype;     /**< The dtype of the result. */
                Core::DataType expected;  /**< The dtype of its parts. */
            };

            /** @brief Returns one row per complex dtype and one real dtype. */
            std::vector<RealDtypeRow> real_dtype_rows()
            {
                return {{"Complex32", Core::DataType::Complex32, Core::DataType::Float32},
                        {"Complex64", Core::DataType::Complex64, Core::DataType::Float64},
                        {"Half", Core::DataType::Half, Core::DataType::Half}};
            }
        } // namespace

        /** @brief Runs one row of real_dtype_rows(). */
        class Small_ScalarResultDtype : public ::testing::TestWithParam<RealDtypeRow>
        {
        };

        TEST_P(Small_ScalarResultDtype, AComplexDtypeHasTheRealDtypeOfItsParts)
        {
            EXPECT_EQ(Internal::real_result_dtype(GetParam().dtype), GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ScalarResultDtype, ::testing::ValuesIn(real_dtype_rows()),
            [](const ::testing::TestParamInfo<RealDtypeRow>& info)
            { return std::string(info.param.name); });

        TEST_F(Medium_ScalarResult, AResultHoldsOneElementOfTheExpectedDtype)
        {
            EXPECT_NO_THROW(Internal::require_scalar_result(engine->create_tensor({1}),
                                                            Core::DataType::Float32, "op"));
            EXPECT_THROW(Internal::require_scalar_result(engine->create_tensor({2}),
                                                         Core::DataType::Float32, "op"),
                         std::invalid_argument);
            EXPECT_THROW(Internal::require_scalar_result(engine->create_tensor({1}),
                                                         Core::DataType::Float64, "op"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
