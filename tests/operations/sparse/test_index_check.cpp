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
 * @file test_index_check.cpp
 * @brief Tests the checks on an index base, a compressed array and an index array.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse/index_check.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds index arrays on the fixture's engine. */
        class Unit_IndexCheck : public SushiBLASTest
        {

        };

        TEST_F(Unit_IndexCheck, OnlyTheTwoEnumeratorsAreIndexBases)
        {
            EXPECT_NO_THROW(Dispatch::require_index_base(Core::IndexBase::One, "op", "A"));
            EXPECT_THROW(Dispatch::require_index_base(Core::IndexBase{}, "op", "A"),
                         std::invalid_argument);
        }

        TEST_F(Unit_IndexCheck, ACompressedArrayIsADenseVectorOfTheExpectedLength)
        {
            const Tensor array = engine->create_tensor({4});

            EXPECT_NO_THROW(Dispatch::require_compressed_array(array, 4, "values", "op", "A"));
            EXPECT_THROW(Dispatch::require_compressed_array(array, 5, "values", "op", "A"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_compressed_array(engine->create_tensor({2, 2}), 4,
                                                            "values", "op", "A"),
                         std::invalid_argument);
        }

        TEST_F(Unit_IndexCheck, AnIndexArrayIsIntegralAndHoldsTheLargestIndex)
        {
            const Tensor narrow = engine->create_tensor({3}, Core::DataType::Int32);
            const Tensor floating = engine->create_tensor({3});
            const int64_t beyond_int32 =
                static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

            EXPECT_NO_THROW(Dispatch::require_index_array(narrow, 3, 100, "indices", "op", "x"));
            EXPECT_THROW(Dispatch::require_index_array(floating, 3, 100, "indices", "op", "x"),
                         std::invalid_argument);
            EXPECT_THROW(
                Dispatch::require_index_array(narrow, 3, beyond_int32, "indices", "op", "x"),
                std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
