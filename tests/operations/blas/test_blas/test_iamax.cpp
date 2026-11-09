/**************************************************************************/
/* test_iamax.cpp                                                         */
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
 * @file test_iamax.cpp
 * @brief Tests that iamax refuses a result tensor that cannot hold the INT64 index.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/operands/scalar_result.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the name of a result dtype for GoogleTest. */
            std::string dtype_name(const ::testing::TestParamInfo<Core::DataType>& info)
            {
                return Core::to_string(info.param);
            }
        } // namespace

        /** @brief Runs the result check of iamax once per dtype that cannot hold the index. */
        class Small_IamaxResult : public ::testing::TestWithParam<Core::DataType>
        {

        };

        /**
         * @brief Tests that iamax refuses a result of another dtype than INT64, which it would
         *        write eight bytes into.
         * @see tests/operations/blas/README.md
         */
        TEST_P(Small_IamaxResult, ADtypeOtherThanInt64IsRefused)
        {
            const std::vector<int64_t> dims = {1};
            const Tensor result = Tensor::make_shape_only(
                SushiRuntime::span<const int64_t>(dims.data(), dims.size()), GetParam());

            EXPECT_THROW(Internal::require_scalar_result_shape(result, Core::DataType::Int64,
                                                               "blas.lvl1.iamax"),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_IamaxResult,
            ::testing::Values(Core::DataType::Float32, Core::DataType::Float64,
                              Core::DataType::Int32, Core::DataType::Half),
            dtype_name);
    } // namespace Test
} // namespace SushiBLAS
