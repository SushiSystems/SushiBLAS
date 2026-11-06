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
 * @brief Tests that BLASOps::iamax returns the zero-based index of the largest magnitude as INT64.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
using namespace SushiBLAS::Test;

class Medium_IAMAXTest : public SushiBLASTest {};

/**
 * @brief Tests that iamax rejects a result tensor too narrow to hold the index.
 * @see tests/operations/blas/README.md
 */
TEST_F(Medium_IAMAXTest, RejectsAResultTensorTooNarrowToHoldTheIndex)
{
    auto X = engine->create_tensor({4});
    fill_tensor(X, {1.0f, -5.0f, 3.0f, 2.0f});

    for (auto dtype : {SushiBLAS::Core::DataType::Float32,
                       SushiBLAS::Core::DataType::Float64,
                       SushiBLAS::Core::DataType::Int32,
                       SushiBLAS::Core::DataType::Half})
    {
        auto R = engine->create_tensor({1}, dtype);
        EXPECT_THROW(SushiBLAS::BLASOps(recorder()).iamax(X, R), std::invalid_argument)
            << "iamax must refuse a " << SushiBLAS::Core::to_string(dtype)
            << " result rather than write an int64_t into it.";
    }
}
