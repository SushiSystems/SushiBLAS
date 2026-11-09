/**************************************************************************/
/* test_reduction_half.cpp                                                */
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
 * @file test_reduction_half.cpp
 * @brief Tests the HALF sum and mean of ReductionOps on the host engine.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @class Small_Level1HalfReductions
         * @brief Records the HALF reductions on a host engine, whose fold runs wider than HALF.
         */
        class Small_Level1HalfReductions : public HostEngineTest
        {

        };

        /**
         * @brief Tests that a HALF mean folds past 65504 before it divides, so the sum does not overflow.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Small_Level1HalfReductions, AMeanFoldsPastTheHalfRangeBeforeItDivides)
        {
            // 60000 is a multiple of 32, the HALF spacing between 32768 and 65504, so it is exact.
            const float K_HALF_ELEMENT = 60000.0f;
            const float K_EXPECTED_MEAN = 60000.0f;

            const std::vector<sycl::half> host = {sycl::half(K_HALF_ELEMENT), sycl::half(K_HALF_ELEMENT)};
            auto t = create_initialized_tensor<sycl::half>({2}, host);
            auto r = engine->create_tensor({1}, Core::DataType::Half);

            SushiBLAS::ReductionOps(recorder()).mean(t, r);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(static_cast<float>(read_tensor<sycl::half>(r)[0]), K_EXPECTED_MEAN);
        }

        /**
         * @brief Tests that a HALF sum of 60000 and 60000 rounds once, to infinity, since 120000 exceeds HALF.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Small_Level1HalfReductions, ASumRoundsOnceIntoTheHalfResult)
        {
            const float K_HALF_ELEMENT = 60000.0f;

            const std::vector<sycl::half> host = {sycl::half(K_HALF_ELEMENT), sycl::half(K_HALF_ELEMENT)};
            auto t = create_initialized_tensor<sycl::half>({2}, host);
            auto r = engine->create_tensor({1}, Core::DataType::Half);

            SushiBLAS::ReductionOps(recorder()).sum(t, r);
            ASSERT_TRUE(recording->run_succeeds());

            const float value = static_cast<float>(read_tensor<sycl::half>(r)[0]);
            EXPECT_TRUE(std::isinf(value) && value > 0.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
