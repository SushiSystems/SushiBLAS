/**************************************************************************/
/* test_scalar_dtype.cpp                                                  */
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
 * @file test_scalar_dtype.cpp
 * @brief Tests that sushiblas_dtype names the DataType of each element type a test uses.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief A C++ element type and the DataType that names it.
        template<typename Element, Core::DataType Expected>
        struct ScalarMapping
        {
            /// @brief The element type.
            using ElementType = Element;
            /// @brief The DataType the element type must map to.
            static constexpr Core::DataType K_EXPECTED = Expected;
        };

        template<typename Mapping>
        class Small_ScalarDtype : public ::testing::Test
        {
        };

        using Mappings = ::testing::Types<
            ScalarMapping<sycl::half, Core::DataType::Half>,
            ScalarMapping<float, Core::DataType::Float32>,
            ScalarMapping<double, Core::DataType::Float64>,
            ScalarMapping<std::complex<float>, Core::DataType::Complex32>,
            ScalarMapping<std::complex<double>, Core::DataType::Complex64>,
            ScalarMapping<int32_t, Core::DataType::Int32>,
            ScalarMapping<int64_t, Core::DataType::Int64>,
            ScalarMapping<uint32_t, Core::DataType::UInt32>>;
        SB_TYPED_TEST_SUITE(Small_ScalarDtype, Mappings);

        TYPED_TEST(Small_ScalarDtype, NamesTheElementType)
        {
            EXPECT_EQ(sushiblas_dtype<typename TypeParam::ElementType>(), TypeParam::K_EXPECTED);
        }
    } // namespace Test
} // namespace SushiBLAS
