/**************************************************************************/
/* test_blas_metadata.cpp                                                 */
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
 * @file test_blas_metadata.cpp
 * @brief Tests that BLASMetadata names the task and fills its slots in the order given.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/scalar.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_BLASMetadata, FillsTheSlotsInTheOrderGiven)
        {
            Internal::BLASMetadata metadata("gemm", SushiRuntime::OperationID());
            metadata.scalar(BLASScalar(std::complex<double>(2.0, -3.0)))
                .flag(Core::Transpose::Trans)
                .extent(int64_t{17});

            const TaskDescription& description = metadata.get();
            EXPECT_EQ(std::strcmp(description.name, "gemm"), 0);
            EXPECT_EQ(description.parameter<double>(0), 2.0);
            EXPECT_EQ(description.parameter<double>(1), -3.0);
            EXPECT_EQ(description.parameter<uint64_t>(2),
                      static_cast<uint64_t>(Core::Transpose::Trans));
            EXPECT_EQ(description.parameter<int64_t>(3), int64_t{17});
            EXPECT_EQ(description.parameter<uint64_t>(4), uint64_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
