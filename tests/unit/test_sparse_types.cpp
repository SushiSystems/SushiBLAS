/**************************************************************************/
/* test_sparse_types.cpp                                                  */
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
 * @file test_sparse_types.cpp
 * @brief Tests IndexBase, the two sparse operands, the kernel index views and accumulator_bytes.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse/accumulator_size.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;
            using Core::IndexBase;

            /** @brief Holds one dtype and the bytes its accumulator is expected to take. */
            struct AccumulatorSize
            {
                DataType dtype;    /**< The element type of a tensor. */
                std::size_t bytes; /**< The expected size of its accumulator. */
            };

            /** @brief Returns a rank-1 tensor of a dtype that owns no storage. */
            [[nodiscard]] Tensor tensor_without_storage(int64_t elements, DataType dtype)
            {
                const int64_t dims[1] = {elements};
                return Tensor(SushiRuntime::sushi_ptr<Storage>(),
                              SushiRuntime::span<const int64_t>(dims, 1), dtype);
            }

            /** @brief Expects a CsrPattern over one-based arrays of Index to read zero-based. */
            template<typename Index>
            void expect_one_based_pattern_reads_zero_based()
            {
                const std::vector<Index> offsets = {1, 3, 3, 6};
                const std::vector<Index> columns = {1, 4, 2, 3, 5};
                const Kernels::CsrPattern<Index> pattern{3, 5, 5, 1, offsets.data(),
                                                         columns.data()};

                const int64_t begins[3] = {0, 2, 2};
                const int64_t ends[3] = {2, 2, 5};
                for (int64_t row = 0; row < 3; ++row)
                {
                    EXPECT_EQ(pattern.row_begin(row), begins[row]) << "row " << row;
                    EXPECT_EQ(pattern.row_end(row), ends[row]) << "row " << row;
                }

                const int64_t zero_based[5] = {0, 3, 1, 2, 4};
                for (int64_t entry = 0; entry < 5; ++entry)
                    EXPECT_EQ(pattern.column(entry), zero_based[entry]) << "entry " << entry;
            }

            /** @brief Expects an IndexList over one-based positions of Index to read zero-based. */
            template<typename Index>
            void expect_one_based_list_reads_zero_based()
            {
                const std::vector<Index> stored = {7, 2, 5};
                const Kernels::IndexList<Index> positions{3, 8, 1, stored.data()};

                const int64_t zero_based[3] = {6, 1, 4};
                for (int64_t entry = 0; entry < 3; ++entry)
                    EXPECT_EQ(positions.at(entry), zero_based[entry]) << "entry " << entry;
            }
        } // namespace

        TEST(Unit_SparseTypes, IndexBaseKeepsItsValuesInOneByte)
        {
            EXPECT_EQ(static_cast<int>(IndexBase::Zero), 221);
            EXPECT_EQ(static_cast<int>(IndexBase::One), 222);
            EXPECT_EQ(sizeof(IndexBase), std::size_t{1});
        }

        TEST(Unit_SparseTypes, IsValidRefusesAValueOutsideTheEnumeration)
        {
            EXPECT_TRUE(Core::is_valid(IndexBase::Zero));
            EXPECT_TRUE(Core::is_valid(IndexBase::One));
            EXPECT_FALSE(Core::is_valid(static_cast<IndexBase>(0)));
            EXPECT_FALSE(Core::is_valid(static_cast<IndexBase>(255)));
        }

        TEST(Unit_SparseTypes, FirstIndexIsTheStoredValueOfPositionZero)
        {
            EXPECT_EQ(Core::first_index(IndexBase::Zero), 0);
            EXPECT_EQ(Core::first_index(IndexBase::One), 1);
        }

        TEST(Unit_SparseTypes, ToStringNamesBothBasesAndAnUnknownValue)
        {
            EXPECT_STREQ(Core::to_string(IndexBase::Zero), "ZERO");
            EXPECT_STREQ(Core::to_string(IndexBase::One), "ONE");
            EXPECT_STREQ(Core::to_string(static_cast<IndexBase>(0)), "UNKNOWN");
        }

        TEST(Unit_SparseTypes, ADefaultSparseVectorIsEmptyAndZeroBased)
        {
            const SparseVector vector;

            EXPECT_EQ(vector.size, 0);
            EXPECT_EQ(vector.nnz(), 0);
            EXPECT_EQ(vector.index_base, IndexBase::Zero);
        }

        TEST(Unit_SparseTypes, ADefaultSparseMatrixIsEmptyAndZeroBased)
        {
            const SparseMatrix matrix;

            EXPECT_EQ(matrix.rows, 0);
            EXPECT_EQ(matrix.cols, 0);
            EXPECT_EQ(matrix.nnz(), 0);
            EXPECT_EQ(matrix.index_base, IndexBase::Zero);
        }

        TEST(Unit_SparseTypes, ASparseVectorReportsTheDtypesOfItsTensors)
        {
            SparseVector vector;
            vector.size = 8;
            vector.indices = tensor_without_storage(3, DataType::Int32);
            vector.values = tensor_without_storage(3, DataType::Complex64);

            EXPECT_EQ(vector.nnz(), 3);
            EXPECT_EQ(vector.dtype(), DataType::Complex64);
            EXPECT_EQ(vector.index_dtype(), DataType::Int32);
        }

        TEST(Unit_SparseTypes, ASparseMatrixReportsTheDtypesOfItsTensors)
        {
            SparseMatrix matrix;
            matrix.rows = 4;
            matrix.cols = 5;
            matrix.row_offsets = tensor_without_storage(5, DataType::Int64);
            matrix.column_indices = tensor_without_storage(7, DataType::Int32);
            matrix.values = tensor_without_storage(7, DataType::Half);

            EXPECT_EQ(matrix.nnz(), 7);
            EXPECT_EQ(matrix.dtype(), DataType::Half);
            EXPECT_EQ(matrix.index_dtype(), DataType::Int64);
        }

        TEST(Unit_SparseTypes, ACsrPatternSubtractsItsFirstIndex)
        {
            expect_one_based_pattern_reads_zero_based<int32_t>();
            expect_one_based_pattern_reads_zero_based<int64_t>();
        }

        TEST(Unit_SparseTypes, AnIndexListSubtractsItsFirstIndex)
        {
            expect_one_based_list_reads_zero_based<int32_t>();
            expect_one_based_list_reads_zero_based<int64_t>();
        }

        TEST(Unit_SparseTypes, AccumulatorBytesIsTheSizeOfTheAccumulatorOfEachDtype)
        {
            const AccumulatorSize sizes[7] = {
                {DataType::Half, 4},      {DataType::Float32, 4},    {DataType::Float64, 8},
                {DataType::Complex32, 8}, {DataType::Complex64, 16}, {DataType::Int32, 0},
                {DataType::Int64, 0}};

            for (const AccumulatorSize& size : sizes)
                EXPECT_EQ(Internal::accumulator_bytes(size.dtype), size.bytes)
                    << Core::to_string(size.dtype);
        }
    } // namespace Test
} // namespace SushiBLAS
