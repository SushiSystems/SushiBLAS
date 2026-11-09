/**************************************************************************/
/* test_integer_arithmetic.cpp                                            */
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
 * @file test_integer_arithmetic.cpp
 * @brief Tests integer elementwise arithmetic, wrap-around included, and double scalars.
 * @author Mustafa Garip
 */

#include <limits>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Runs the cases of neg and abs on the smallest integer of one element type.
         * @tparam Int int32_t or int64_t.
         */
        template<typename Int>
        class Small_IntegerArithmeticTest : public HostEngineTest
        {
            protected:
                /// @brief The smallest value of Int.
                static constexpr Int K_MIN = std::numeric_limits<Int>::min();
        };

        /// @brief The integer element types under test.
        using IntegerTypes = ::testing::Types<int32_t, int64_t>;
        TYPED_TEST_SUITE(Small_IntegerArithmeticTest, IntegerTypes, DefaultTypeNames);

        /**
         * @brief Tests that neg maps MIN to itself and negates the other values.
         */
        TYPED_TEST(Small_IntegerArithmeticTest, NegMapsTheMinimumToItself)
        {
            using Int = TypeParam;
            auto n = this->template create_initialized_tensor<Int>({3}, {5, -3, TestFixture::K_MIN});

            SushiBLAS::ElementwiseOps(this->recorder()).neg(n);
            ASSERT_TRUE(this->recording->run_succeeds());

            EXPECT_EQ((std::vector<Int>{-5, 3, TestFixture::K_MIN}), read_tensor<Int>(n));
        }

        /**
         * @brief Tests that abs maps MIN to itself and takes the magnitude of the other values.
         */
        TYPED_TEST(Small_IntegerArithmeticTest, AbsMapsTheMinimumToItself)
        {
            using Int = TypeParam;
            auto m = this->template create_initialized_tensor<Int>({3}, {5, -3, TestFixture::K_MIN});

            SushiBLAS::ElementwiseOps(this->recorder()).abs(m);
            ASSERT_TRUE(this->recording->run_succeeds());

            EXPECT_EQ((std::vector<Int>{5, 3, TestFixture::K_MIN}), read_tensor<Int>(m));
        }
    } // namespace Test
} // namespace SushiBLAS
