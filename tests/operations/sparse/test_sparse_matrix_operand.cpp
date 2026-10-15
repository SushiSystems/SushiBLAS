/**************************************************************************/
/* test_sparse_matrix_operand.cpp                                         */
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
 * @file test_sparse_matrix_operand.cpp
 * @brief Tests how a CSR matrix is described and the shapes of its arrays that are refused.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/sparse/sparse_matrix_operand.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds CSR matrices on the fixture's engine. */
        class Unit_SparseMatrixOperand : public SushiBLASTest
        {
            protected:
                /** @brief Returns a 3 x 4 one-based matrix of five entries with INT32 indices. */
                [[nodiscard]] SparseMatrix three_by_four() const
                {
                    SparseMatrix a;
                    a.rows = 3;
                    a.cols = 4;
                    a.index_base = Core::IndexBase::One;
                    a.row_offsets = engine->create_tensor({4}, Core::DataType::Int32);
                    a.column_indices = engine->create_tensor({5}, Core::DataType::Int32);
                    a.values = engine->create_tensor({5});
                    return a;
                }
        };

        TEST_F(Unit_SparseMatrixOperand, DescribesItsExtentsEntriesAndFirstIndex)
        {
            const Dispatch::SparseMatrixOperand m =
                Dispatch::describe_sparse_matrix(three_by_four(), "op", "A");

            EXPECT_EQ(m.rows, int64_t{3});
            EXPECT_EQ(m.cols, int64_t{4});
            EXPECT_EQ(m.nnz, int64_t{5});
            EXPECT_EQ(m.first_index, int64_t{1});
        }

        TEST_F(Unit_SparseMatrixOperand, RefusesAWrongOffsetCountAndMixedIndexDtypes)
        {
            SparseMatrix short_offsets = three_by_four();
            short_offsets.row_offsets = engine->create_tensor({3}, Core::DataType::Int32);
            SparseMatrix mixed = three_by_four();
            mixed.column_indices = engine->create_tensor({5}, Core::DataType::Int64);

            EXPECT_THROW((void)Dispatch::describe_sparse_matrix(short_offsets, "op", "A"),
                         std::invalid_argument);
            EXPECT_THROW((void)Dispatch::describe_sparse_matrix(mixed, "op", "A"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
