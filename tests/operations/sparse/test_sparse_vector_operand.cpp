/**************************************************************************/
/* test_sparse_vector_operand.cpp                                         */
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
 * @file test_sparse_vector_operand.cpp
 * @brief Tests how a compressed vector is described and the sizes that are refused.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/sparse/sparse_vector_operand.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds compressed vectors on the fixture's engine. */
        class Medium_SparseVectorOperand : public SushiBLASTest
        {
            protected:
                /** @brief Returns a zero-based vector of a size storing a count of entries. */
                [[nodiscard]] SparseVector compressed(int64_t size, int64_t entries) const
                {
                    SparseVector x;
                    x.size = size;
                    x.indices = engine->create_tensor({entries}, Core::DataType::Int64);
                    x.values = engine->create_tensor({entries});
                    return x;
                }
        };

        TEST_F(Medium_SparseVectorOperand, DescribesItsSizeEntriesAndFirstIndex)
        {
            const Dispatch::SparseVectorOperand v =
                Dispatch::describe_sparse_vector(compressed(10, 3), "op", "x");

            EXPECT_EQ(v.size, int64_t{10});
            EXPECT_EQ(v.nnz, int64_t{3});
            EXPECT_EQ(v.first_index, int64_t{0});
        }

        TEST_F(Medium_SparseVectorOperand, RefusesANegativeSizeAndMoreEntriesThanPositions)
        {
            EXPECT_THROW((void)Dispatch::describe_sparse_vector(compressed(-1, 0), "op", "x"),
                         std::invalid_argument);
            EXPECT_THROW((void)Dispatch::describe_sparse_vector(compressed(2, 3), "op", "x"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
