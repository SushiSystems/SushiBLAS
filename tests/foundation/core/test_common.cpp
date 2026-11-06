/**************************************************************************/
/* test_common.cpp                                                        */
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
 * @file test_common.cpp
 * @brief Tests the names, the element sizes and the kind predicates of DataType, and Layout.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(static_cast<int>(Core::DataType::Half) == 0, "HALF is 0.");
        static_assert(static_cast<int>(Core::DataType::Float32) == 1, "FLOAT32 is 1.");
        static_assert(static_cast<int>(Core::DataType::Float64) == 2, "FLOAT64 is 2.");
        static_assert(static_cast<int>(Core::DataType::Complex32) == 3, "COMPLEX32 is 3.");
        static_assert(static_cast<int>(Core::DataType::Complex64) == 4, "COMPLEX64 is 4.");
        static_assert(static_cast<int>(Core::DataType::Int32) == 5, "INT32 is 5.");
        static_assert(static_cast<int>(Core::DataType::Int64) == 6, "INT64 is 6.");
        static_assert(static_cast<int>(Core::DataType::UInt32) == 7, "UINT32 is 7.");
        static_assert(Core::K_DATA_TYPE_COUNT == 8, "DataType has eight enumerators.");
        static_assert(sizeof(Core::DataType) == 1, "DataType is one byte.");
        static_assert(sizeof(Core::Layout) == 1, "Layout is one byte.");

        /// @brief A value outside every DataType enumerator.
        constexpr int K_UNKNOWN_DTYPE = 200;

        TEST(Small_DataType, EveryEnumeratorKeepsItsFileFormatNumber)
        {
            EXPECT_EQ(static_cast<int>(Core::DataType::Half), 0);
            EXPECT_EQ(static_cast<int>(Core::DataType::Float32), 1);
            EXPECT_EQ(static_cast<int>(Core::DataType::Float64), 2);
            EXPECT_EQ(static_cast<int>(Core::DataType::Complex32), 3);
            EXPECT_EQ(static_cast<int>(Core::DataType::Complex64), 4);
            EXPECT_EQ(static_cast<int>(Core::DataType::Int32), 5);
            EXPECT_EQ(static_cast<int>(Core::DataType::Int64), 6);
            EXPECT_EQ(static_cast<int>(Core::DataType::UInt32), 7);
        }

        TEST(Small_DataType, ToStringNamesEveryEnumerator)
        {
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::Half)), "HALF");
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::Float32)), "FLOAT32");
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::Float64)), "FLOAT64");
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::Complex32)), "COMPLEX32");
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::Complex64)), "COMPLEX64");
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::Int32)), "INT32");
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::Int64)), "INT64");
            EXPECT_EQ(std::string(Core::to_string(Core::DataType::UInt32)), "UINT32");
        }

        TEST(Small_DataType, ToStringNamesAValueOutsideTheEnumerationUnknown)
        {
            const auto unknown = static_cast<Core::DataType>(K_UNKNOWN_DTYPE);

            EXPECT_EQ(std::string(Core::to_string(unknown)), "UNKNOWN");
        }

        TEST(Small_DataType, BytesPerElementIsTheStorageSizeOfEachType)
        {
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::Half), 2u);
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::Float32), 4u);
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::Float64), 8u);
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::Complex32), 8u);
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::Complex64), 16u);
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::Int32), 4u);
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::Int64), 8u);
            EXPECT_EQ(Core::bytes_per_element(Core::DataType::UInt32), 4u);
        }

        /// @brief Names a computed type by its to_string, which gtest takes as the test's suffix.
        static std::string dtype_name(const ::testing::TestParamInfo<Core::DataType>& info)
        {
            return Core::to_string(info.param);
        }

        /// @brief Returns every computed type, which is every DataType but UInt32.
        static std::vector<Core::DataType> computed_types()
        {
            return {Core::DataType::Half,      Core::DataType::Float32,   Core::DataType::Float64,
                    Core::DataType::Complex32, Core::DataType::Complex64, Core::DataType::Int32,
                    Core::DataType::Int64};
        }

        /// @brief Runs the kind predicates on one computed type.
        class Small_DataTypeKind : public ::testing::TestWithParam<Core::DataType>
        {
        };

        TEST_P(Small_DataTypeKind, ATypeIsExactlyOneOfRealComplexAndInteger)
        {
            const Core::DataType dtype = GetParam();
            const int kinds = static_cast<int>(Core::is_real_dtype(dtype)) +
                              static_cast<int>(Core::is_complex_dtype(dtype)) +
                              static_cast<int>(Core::is_integer_dtype(dtype));

            EXPECT_EQ(kinds, 1);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_DataTypeKind, ::testing::ValuesIn(computed_types()),
                                 dtype_name);

        TEST(Small_DataType, ThePredicatesSortEachTypeIntoItsKind)
        {
            EXPECT_TRUE(Core::is_real_dtype(Core::DataType::Half));
            EXPECT_TRUE(Core::is_real_dtype(Core::DataType::Float32));
            EXPECT_TRUE(Core::is_real_dtype(Core::DataType::Float64));
            EXPECT_TRUE(Core::is_complex_dtype(Core::DataType::Complex32));
            EXPECT_TRUE(Core::is_complex_dtype(Core::DataType::Complex64));
            EXPECT_TRUE(Core::is_integer_dtype(Core::DataType::Int32));
            EXPECT_TRUE(Core::is_integer_dtype(Core::DataType::Int64));
        }

        TEST(Small_DataType, UInt32IsNeitherRealNorComplexNorSignedInteger)
        {
            EXPECT_FALSE(Core::is_real_dtype(Core::DataType::UInt32));
            EXPECT_FALSE(Core::is_complex_dtype(Core::DataType::UInt32));
            EXPECT_FALSE(Core::is_integer_dtype(Core::DataType::UInt32));
        }

        TEST(Small_Layout, KeepsItsNumbers)
        {
            EXPECT_EQ(static_cast<int>(Core::Layout::RowMajor), 0);
            EXPECT_EQ(static_cast<int>(Core::Layout::ColumnMajor), 1);
        }
    } // namespace Test
} // namespace SushiBLAS
