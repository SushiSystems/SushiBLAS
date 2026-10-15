/**************************************************************************/
/* test_sparse_vector.cpp                                                 */
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
 * @file test_sparse_vector.cpp
 * @brief Tests that a SparseVector reads its entry count and both dtypes from its tensors.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a rank-one tensor of count elements of dtype.
        static Tensor entries_of(int64_t count, Core::DataType dtype)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const int64_t dims[1] = {count};
            return factory.create(SushiRuntime::span<const int64_t>(dims, 1), dtype);
        }

        TEST(Unit_SparseVector, ADefaultVectorIsEmptyAndZeroBased)
        {
            const SparseVector vector;

            EXPECT_EQ(vector.size, int64_t{0});
            EXPECT_EQ(vector.index_base, Core::IndexBase::Zero);
            EXPECT_EQ(vector.nnz(), int64_t{0});
        }

        TEST(Unit_SparseVector, ReadsItsCountAndDtypesFromItsTensors)
        {
            SparseVector vector;
            vector.size = 10;
            vector.index_base = Core::IndexBase::One;
            vector.indices = entries_of(3, Core::DataType::Int64);
            vector.values = entries_of(3, Core::DataType::Half);

            EXPECT_EQ(vector.nnz(), int64_t{3});
            EXPECT_EQ(vector.dtype(), Core::DataType::Half);
            EXPECT_EQ(vector.index_dtype(), Core::DataType::Int64);
        }
    } // namespace Test
} // namespace SushiBLAS
