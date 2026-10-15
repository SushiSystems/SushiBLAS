/**************************************************************************/
/* test_sparse_matrix.cpp                                                 */
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
 * @file test_sparse_matrix.cpp
 * @brief Tests that a SparseMatrix reads its entry count and both dtypes from its tensors.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a rank-one tensor of count elements of dtype.
        static Tensor vector_of(int64_t count, Core::DataType dtype)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const int64_t dims[1] = {count};
            return factory.create(SushiRuntime::span<const int64_t>(dims, 1), dtype);
        }

        TEST(Unit_SparseMatrix, ADefaultMatrixIsEmptyAndZeroBased)
        {
            const SparseMatrix matrix;

            EXPECT_EQ(matrix.rows, int64_t{0});
            EXPECT_EQ(matrix.cols, int64_t{0});
            EXPECT_EQ(matrix.index_base, Core::IndexBase::Zero);
            EXPECT_EQ(matrix.nnz(), int64_t{0});
        }

        TEST(Unit_SparseMatrix, ReadsItsCountAndDtypesFromItsTensors)
        {
            SparseMatrix matrix;
            matrix.rows = 3;
            matrix.cols = 4;
            matrix.row_offsets = vector_of(4, Core::DataType::Int32);
            matrix.column_indices = vector_of(5, Core::DataType::Int32);
            matrix.values = vector_of(5, Core::DataType::Complex64);

            EXPECT_EQ(matrix.nnz(), int64_t{5});
            EXPECT_EQ(matrix.dtype(), Core::DataType::Complex64);
            EXPECT_EQ(matrix.index_dtype(), Core::DataType::Int32);
        }
    } // namespace Test
} // namespace SushiBLAS
