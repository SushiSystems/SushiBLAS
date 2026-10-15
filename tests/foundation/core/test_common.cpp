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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_common.cpp
 * @brief Tests the names, the element sizes and the kind predicates of DataType, and Layout.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief A value outside every DataType enumerator.
        constexpr int K_UNKNOWN_DTYPE = 200;

        TEST(Small_DataType, EveryEnumeratorKeepsItsFileFormatNumber)
        {
            static_assert(static_cast<int>(Core::DataType::Half) == 0, "HALF is 0.");
            static_assert(static_cast<int>(Core::DataType::Float32) == 1, "FLOAT32 is 1.");
            static_assert(static_cast<int>(Core::DataType::Float64) == 2, "FLOAT64 is 2.");
            static_assert(static_cast<int>(Core::DataType::Complex32) == 3, "COMPLEX32 is 3.");
            static_assert(static_cast<int>(Core::DataType::Complex64) == 4, "COMPLEX64 is 4.");
            static_assert(static_cast<int>(Core::DataType::Int32) == 5, "INT32 is 5.");
            static_assert(static_cast<int>(Core::DataType::Int64) == 6, "INT64 is 6.");

            EXPECT_EQ(static_cast<int>(Core::DataType::Half), 0);
            EXPECT_EQ(static_cast<int>(Core::DataType::Float32), 1);
            EXPECT_EQ(static_cast<int>(Core::DataType::Float64), 2);
            EXPECT_EQ(static_cast<int>(Core::DataType::Complex32), 3);
            EXPECT_EQ(static_cast<int>(Core::DataType::Complex64), 4);
            EXPECT_EQ(static_cast<int>(Core::DataType::Int32), 5);
            EXPECT_EQ(static_cast<int>(Core::DataType::Int64), 6);
        }

        TEST(Small_DataType, IsOneByte)
        {
            static_assert(sizeof(Core::DataType) == 1, "DataType is one byte.");

            EXPECT_EQ(sizeof(Core::DataType), 1u);
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
        }

        TEST(Small_DataType, EveryTypeIsExactlyOneOfRealComplexAndInteger)
        {
            const Core::DataType all[] = {
                Core::DataType::Half,      Core::DataType::Float32,   Core::DataType::Float64,
                Core::DataType::Complex32, Core::DataType::Complex64, Core::DataType::Int32,
                Core::DataType::Int64};

            for (Core::DataType dtype : all)
            {
                SCOPED_TRACE(Core::to_string(dtype));
                const int kinds = static_cast<int>(Core::is_real_dtype(dtype)) +
                                  static_cast<int>(Core::is_complex_dtype(dtype)) +
                                  static_cast<int>(Core::is_integer_dtype(dtype));
                EXPECT_EQ(kinds, 1);
            }
        }

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

        TEST(Small_Layout, KeepsItsNumbersAndIsOneByte)
        {
            static_assert(sizeof(Core::Layout) == 1, "Layout is one byte.");

            EXPECT_EQ(static_cast<int>(Core::Layout::RowMajor), 0);
            EXPECT_EQ(static_cast<int>(Core::Layout::ColumnMajor), 1);
        }

        TEST(Small_Layout, TheLargestRankIsSix)
        {
            EXPECT_EQ(Core::K_MAX_TENSOR_RANK, std::size_t{6});
        }
    } // namespace Test
} // namespace SushiBLAS
