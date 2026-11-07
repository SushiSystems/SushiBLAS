/**************************************************************************/
/* test_index_check.cpp                                                   */
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
 * @file test_index_check.cpp
 * @brief Tests the checks on an index base, a compressed array and an index array on arrays
 *        built by hand, so no runtime runs.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <functional>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse/index_check.hpp>
#include "support/hand_built_operand.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The made-up allocation of the array every row checks.
            constexpr std::size_t K_ARRAY_ALLOCATION = 1;

            /// @brief The largest value an INT32 index array holds.
            constexpr int64_t K_INT32_LARGEST = std::numeric_limits<int32_t>::max();

            /**
             * @brief Expects a check to accept its input when accepted is set and to refuse it with
             *        std::invalid_argument otherwise.
             */
            void expect_verdict(bool accepted, const std::function<void()>& check)
            {
                if (accepted)
                    EXPECT_NO_THROW(check());
                else
                    EXPECT_THROW(check(), std::invalid_argument);
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            template<typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }

            /** @brief Holds one value of the index base and whether the check accepts it. */
            struct BaseRow
            {
                const char* name;      /**< The row's suffix. */
                Core::IndexBase base;  /**< The value handed to the check. */
                bool accepted;         /**< Whether the value is an enumerator. */
            };

            /** @brief Returns both enumerators, the value-initialised base and a value past them. */
            std::vector<BaseRow> base_rows()
            {
                return {{"Zero", Core::IndexBase::Zero, true},
                        {"One", Core::IndexBase::One, true},
                        {"ValueInitialised", Core::IndexBase{}, false},
                        {"PastTheLastEnumerator", static_cast<Core::IndexBase>(255), false}};
            }

            /** @brief Holds the shape of one array handed to require_compressed_array. */
            struct ArrayRow
            {
                const char* name;               /**< The row's suffix. */
                std::vector<int64_t> dims;      /**< The extents of the array. */
                int64_t expected;               /**< The length the operand needs. */
                int64_t stride;                 /**< The stride of dimension 0; 0 keeps the dense one. */
                bool has_storage;               /**< Whether the array has storage. */
                bool accepted;                  /**< Whether the check accepts the array. */
            };

            /** @brief Returns the arrays a compressed-array check accepts and each rule it refuses. */
            std::vector<ArrayRow> array_rows()
            {
                return {{"ADenseVectorOfTheExpectedLength", {4}, 4, 0, true, true},
                        {"AnotherLength", {4}, 5, 0, true, false},
                        {"AnotherRank", {2, 2}, 4, 0, true, false},
                        {"AStridedVector", {4}, 4, 2, true, false},
                        {"NoStorage", {4}, 4, 0, false, false},
                        {"AnEmptyVectorWithoutStorage", {0}, 0, 0, false, true}};
            }

            /** @brief Returns the hand-built array a row describes. */
            HandBuiltOperand array_of(const ArrayRow& row)
            {
                HandBuiltOperand array(row.dims, Core::DataType::Int32, K_ARRAY_ALLOCATION);
                if (row.stride != 0)
                    array.with_stride(0, row.stride);
                if (!row.has_storage)
                    array.without_storage();
                return array;
            }

            /** @brief Holds the dtype of one index array and whether the check accepts it. */
            struct DtypeRow
            {
                const char* name;       /**< The row's suffix. */
                Core::DataType dtype;   /**< The element type of the array. */
                bool accepted;          /**< Whether it is an integer dtype of an index. */
            };

            /** @brief Returns the dtypes the check accepts and one that is floating point. */
            std::vector<DtypeRow> dtype_rows()
            {
                return {{"Int32", Core::DataType::Int32, true},
                        {"Int64", Core::DataType::Int64, true},
                        {"Float32", Core::DataType::Float32, false}};
            }

            /** @brief Holds an index dtype, the largest index it must hold and the verdict. */
            struct LargestRow
            {
                const char* name;       /**< The row's suffix. */
                Core::DataType dtype;   /**< The element type of the array. */
                int64_t largest;        /**< The largest index the operand stores. */
                bool accepted;          /**< Whether the dtype can hold it. */
            };

            /** @brief Returns the largest indices on both sides of the INT32 limit. */
            std::vector<LargestRow> largest_rows()
            {
                return {{"Int32HoldingAHundred", Core::DataType::Int32, 100, true},
                        {"Int32HoldingItsLimit", Core::DataType::Int32, K_INT32_LARGEST, true},
                        {"Int32BeyondItsLimit", Core::DataType::Int32, K_INT32_LARGEST + 1, false},
                        {"Int64BeyondTheInt32Limit", Core::DataType::Int64, K_INT32_LARGEST + 1,
                         true}};
            }
        } // namespace

        /** @brief Checks one index base at a time. */
        class Small_RequireIndexBase : public ::testing::TestWithParam<BaseRow>
        {

        };

        TEST_P(Small_RequireIndexBase, AcceptsTheTwoEnumeratorsAndNothingElse)
        {
            const BaseRow& row = GetParam();

            expect_verdict(row.accepted,
                           [&] { Dispatch::require_index_base(row.base, "op", "A"); });
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RequireIndexBase, ::testing::ValuesIn(base_rows()),
                                 row_name<BaseRow>);

        /** @brief Checks one array at a time. */
        class Small_RequireCompressedArray : public ::testing::TestWithParam<ArrayRow>
        {

        };

        TEST_P(Small_RequireCompressedArray, AcceptsADenseVectorOfTheExpectedLengthAndNothingElse)
        {
            const ArrayRow& row = GetParam();
            const HandBuiltOperand array = array_of(row);

            expect_verdict(row.accepted, [&] {
                Dispatch::require_compressed_array(array.placed(), row.expected, "values", "op",
                                                   "A");
            });
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RequireCompressedArray, ::testing::ValuesIn(array_rows()),
                                 row_name<ArrayRow>);

        /** @brief Checks the dtype of one index array at a time. */
        class Small_RequireIndexArrayDtype : public ::testing::TestWithParam<DtypeRow>
        {

        };

        TEST_P(Small_RequireIndexArrayDtype, AcceptsAnIntegerDtypeAndNothingElse)
        {
            const DtypeRow& row = GetParam();
            const HandBuiltOperand array({3}, row.dtype, K_ARRAY_ALLOCATION);

            expect_verdict(row.accepted, [&] {
                Dispatch::require_index_array(array.placed(), 3, 100, "indices", "op", "x");
            });
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RequireIndexArrayDtype, ::testing::ValuesIn(dtype_rows()),
                                 row_name<DtypeRow>);

        /** @brief Checks the largest index against one index dtype at a time. */
        class Small_RequireIndexArrayLargest : public ::testing::TestWithParam<LargestRow>
        {

        };

        TEST_P(Small_RequireIndexArrayLargest, RefusesAnIndexBeyondTheLimitOfItsDtype)
        {
            const LargestRow& row = GetParam();
            const HandBuiltOperand array({3}, row.dtype, K_ARRAY_ALLOCATION);

            expect_verdict(row.accepted, [&] {
                Dispatch::require_index_array(array.placed(), 3, row.largest, "indices", "op",
                                              "x");
            });
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RequireIndexArrayLargest,
                                 ::testing::ValuesIn(largest_rows()), row_name<LargestRow>);
    } // namespace Test
} // namespace SushiBLAS
