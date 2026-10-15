/**************************************************************************/
/* test_sparse_builder.cpp                                                */
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
 * @file test_sparse_builder.cpp
 * @brief Tests the sparse vectors and CSR matrices SparseBuilder makes from host data.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/sparse_storage/sparse_builder.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>
#include <SushiBLAS/sparse_kernels/reference/structure.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;
            using Core::IndexBase;
            using Core::Layout;
            using Kernels::Reference::find_defect;
            using Kernels::Reference::StructureDefect;

            /** @brief A host vector of expectations or of operand values. */
            using Wide = std::vector<std::complex<double>>;

            /// @brief The two layouts a dense tensor is compressed from.
            constexpr Layout K_BOTH_LAYOUTS[2] = {Layout::RowMajor, Layout::ColumnMajor};

            /// @brief Length of the dense vector the sparse vectors of this file compress.
            constexpr int64_t K_VECTOR_SIZE = 8;

            /// @brief Rows of the dense tensor that takes more than one transfer: 24 MB of FLOAT32.
            constexpr int64_t K_LARGE_ROWS = 3000;

            /// @brief Columns of that tensor.
            constexpr int64_t K_LARGE_COLS = 2000;

            /// @brief Element (i, j) is 1 where i + K_LARGE_WEIGHT * j is a multiple of this.
            constexpr int64_t K_LARGE_MODULUS = 97;

            /// @brief The weight of the column in that rule.
            constexpr int64_t K_LARGE_WEIGHT = 3;

            /** @brief Expects a matrix to hold zero-based CSR arrays, stored in Index with base. */
            template<typename T, typename Index>
            void expect_matrix(const SparseMatrix& A, const std::vector<int64_t>& offsets,
                               const std::vector<int64_t>& columns, const Wide& values,
                               IndexBase base)
            {
                EXPECT_EQ(A.index_base, base);
                ASSERT_EQ(A.index_dtype(), sushiblas_dtype<Index>());
                ASSERT_EQ(A.dtype(), sushiblas_dtype<T>());
                ASSERT_EQ(A.column_indices.dtype, sushiblas_dtype<Index>());
                ASSERT_EQ(A.nnz(), static_cast<int64_t>(columns.size()));
                ASSERT_EQ(A.column_indices.num_elements, A.nnz());

                EXPECT_EQ(read_tensor<Index>(A.row_offsets), stored_indices<Index>(offsets, base));
                if (A.nnz() == 0)
                    return;

                EXPECT_EQ(read_tensor<Index>(A.column_indices),
                          stored_indices<Index>(columns, base));
                const std::vector<T> actual = read_tensor<T>(A.values);
                for (std::size_t k = 0; k < actual.size(); ++k)
                    EXPECT_EQ(widened(actual[k]), values[k]) << "entry " << k;

                const Kernels::CsrPattern<Index> pattern{
                    A.rows, A.cols, A.nnz(), Core::first_index(base),
                    A.row_offsets.data_as<Index>(), A.column_indices.data_as<Index>()};
                EXPECT_EQ(find_defect(pattern), StructureDefect::None);
            }

            /** @brief Expects a matrix to hold the 4 x 5 sample, stored in Index with base. */
            template<typename T, typename Index>
            void expect_sample(const SparseMatrix& A, IndexBase base)
            {
                SCOPED_TRACE(Core::to_string(base));
                EXPECT_EQ(A.rows, 4);
                EXPECT_EQ(A.cols, 5);
                expect_matrix<T, Index>(A, sample_row_offsets(), sample_columns(),
                                        sample_values<T>(), base);
            }

            /** @brief Returns the sample matrix as the 20 elements of a dense one, row by row. */
            template<typename T>
            [[nodiscard]] Wide sample_by_rows()
            {
                const HostTriplets<T> host = sample_matrix<T>();
                Wide by_rows(static_cast<std::size_t>(host.rows * host.cols));
                for (std::size_t k = 0; k < host.values.size(); ++k)
                    by_rows[static_cast<std::size_t>(host.row_indices[k] * host.cols +
                                                     host.column_indices[k])] =
                        widened(host.values[k]);
                return by_rows;
            }
        } // namespace

        /**
         * @brief Builds sparse operands and dense tensors of one element and one index type.
         */
        template<typename TypeParam>
        class Unit_SparseBuilder : public SushiBLASTest
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief Returns the CSR matrix of host triplets, stored with base. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<T>& host, IndexBase base)
                {
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<Index>(host.row_indices, base),
                        stored_indices<Index>(host.column_indices, base), host.values, base);
                }

                /** @brief Returns the sparse vector of zero-based positions, stored with base. */
                [[nodiscard]] SparseVector build_vector(const std::vector<int64_t>& positions,
                                                        const Wide& values, IndexBase base)
                {
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.vector_from_entries(K_VECTOR_SIZE,
                                                       stored_indices<Index>(positions, base),
                                                       narrowed_values(values), base);
                }

                /** @brief Returns the CSR matrix of a dense tensor, with indices of index_dtype. */
                [[nodiscard]] SparseMatrix compress(const Tensor& dense, DataType index_dtype,
                                                    IndexBase base)
                {
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.matrix_from_dense(dense, index_dtype, base);
                }

                /**
                 * @brief Returns a dense tensor of the engine's layout holding a matrix.
                 * @param by_rows rows * cols elements, row by row; neither extent is 0.
                 */
                [[nodiscard]] Tensor dense_tensor(int64_t rows, int64_t cols, const Wide& by_rows)
                {
                    const bool row_major = engine->layout() == Layout::RowMajor;
                    std::vector<T> elements(by_rows.size());
                    for (int64_t i = 0; i < rows; ++i)
                    {
                        for (int64_t j = 0; j < cols; ++j)
                        {
                            const int64_t source = i * cols + j;
                            const int64_t target = row_major ? source : i + j * rows;
                            elements[static_cast<std::size_t>(target)] =
                                narrowed<T>(by_rows[static_cast<std::size_t>(source)]);
                        }
                    }
                    return create_initialized_tensor<T>({rows, cols}, elements);
                }

                /** @brief Returns a dense tensor of T that holds no element. */
                [[nodiscard]] Tensor dense_without_elements(int64_t rows, int64_t cols)
                {
                    return engine->create_tensor({rows, cols}, sushiblas_dtype<T>());
                }

                /** @brief Returns host expectations narrowed to T. */
                [[nodiscard]] static std::vector<T> narrowed_values(const Wide& values)
                {
                    std::vector<T> host(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        host[i] = narrowed<T>(values[i]);
                    return host;
                }
        };

        SB_TYPED_TEST_SUITE(Unit_SparseBuilder, SparseCases);

        TYPED_TEST(Unit_SparseBuilder, TheSampleTripletsGiveTheSampleMatrixUnderBothBases)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            for (const IndexBase base : K_BOTH_BASES)
                expect_sample<T, Index>(this->build(sample_matrix<T>(), base), base);
        }

        TYPED_TEST(Unit_SparseBuilder, TheThreeTensorsHaveRankOneAndTheDtypesOfTheirHostTypes)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            const SparseMatrix A = this->build(sample_matrix<T>(), IndexBase::Zero);

            EXPECT_EQ(A.row_offsets.rank, 1);
            EXPECT_EQ(A.column_indices.rank, 1);
            EXPECT_EQ(A.values.rank, 1);
            EXPECT_EQ(A.row_offsets.num_elements, 5);
            EXPECT_EQ(A.column_indices.num_elements, 7);
            EXPECT_EQ(A.values.num_elements, 7);
            EXPECT_EQ(A.row_offsets.dtype, sushiblas_dtype<Index>());
            EXPECT_EQ(A.column_indices.dtype, sushiblas_dtype<Index>());
            EXPECT_EQ(A.values.dtype, sushiblas_dtype<T>());
        }

        TYPED_TEST(Unit_SparseBuilder, EntriesGivenTwiceAsHalvesGiveTheSameSevenEntries)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            const HostTriplets<T> whole = sample_matrix<T>();
            HostTriplets<T> halves;
            halves.rows = whole.rows;
            halves.cols = whole.cols;
            for (int pass = 0; pass < 2; ++pass)
            {
                for (std::size_t k = 0; k < whole.values.size(); ++k)
                {
                    halves.row_indices.push_back(whole.row_indices[k]);
                    halves.column_indices.push_back(whole.column_indices[k]);
                    halves.values.push_back(narrowed<T>(widened(whole.values[k]) * 0.5));
                }
            }

            for (const IndexBase base : K_BOTH_BASES)
                expect_sample<T, Index>(this->build(halves, base), base);
        }

        TYPED_TEST(Unit_SparseBuilder, AMatrixWithoutEntriesHasNoValueStorage)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            HostTriplets<T> host;
            host.rows = 3;
            host.cols = 4;

            for (const IndexBase base : K_BOTH_BASES)
            {
                const SparseMatrix A = this->build(host, base);

                EXPECT_EQ(A.nnz(), 0);
                EXPECT_EQ(A.values.rank, 1);
                EXPECT_TRUE(A.values.storage == nullptr);
                EXPECT_TRUE(A.column_indices.storage == nullptr);
                EXPECT_EQ(read_tensor<Index>(A.row_offsets),
                          std::vector<Index>(4, static_cast<Index>(Core::first_index(base))));
            }
        }

        TYPED_TEST(Unit_SparseBuilder, ADenseTensorCompressesToTheSampleInEitherLayout)
        {
            using T = typename TypeParam::Value;

            for (const Layout layout : K_BOTH_LAYOUTS)
            {
                SCOPED_TRACE(layout == Layout::RowMajor ? "row-major" : "column-major");
                this->reinit_engine(layout);
                const Tensor dense = this->dense_tensor(4, 5, sample_by_rows<T>());

                for (const IndexBase base : K_BOTH_BASES)
                {
                    expect_sample<T, int32_t>(this->compress(dense, DataType::Int32, base), base);
                    expect_sample<T, int64_t>(this->compress(dense, DataType::Int64, base), base);
                }
            }
        }

        TYPED_TEST(Unit_SparseBuilder, ADenseTensorDropsANegativeZeroAndKeepsANaN)
        {
            using T = typename TypeParam::Value;

            const std::complex<double> negative_zero(-0.0, -0.0);
            const std::complex<double> nan(std::nan(""), 0.0);
            const Wide by_rows = {negative_zero, nan, {0.0, 0.0}, {3.0, 0.0}};

            for (const Layout layout : K_BOTH_LAYOUTS)
            {
                this->reinit_engine(layout);
                const SparseMatrix A = this->compress(this->dense_tensor(2, 2, by_rows),
                                                      DataType::Int64, IndexBase::Zero);

                ASSERT_EQ(A.nnz(), 2);
                EXPECT_EQ(read_tensor<int64_t>(A.row_offsets), (std::vector<int64_t>{0, 1, 2}));
                EXPECT_EQ(read_tensor<int64_t>(A.column_indices), (std::vector<int64_t>{1, 1}));

                const std::vector<T> values = read_tensor<T>(A.values);
                EXPECT_TRUE(std::isnan(widened(values[0]).real()));
                EXPECT_EQ(widened(values[1]), std::complex<double>(3.0, 0.0));
            }
        }

        TYPED_TEST(Unit_SparseBuilder, ADenseTensorWithoutElementsGivesAMatrixWithoutEntries)
        {
            const SparseMatrix no_rows = this->compress(this->dense_without_elements(0, 5),
                                                        DataType::Int64, IndexBase::One);
            EXPECT_EQ(no_rows.rows, 0);
            EXPECT_EQ(no_rows.cols, 5);
            EXPECT_EQ(no_rows.nnz(), 0);
            EXPECT_EQ(read_tensor<int64_t>(no_rows.row_offsets), std::vector<int64_t>(1, 1));

            const SparseMatrix no_columns = this->compress(this->dense_without_elements(5, 0),
                                                           DataType::Int64, IndexBase::One);
            EXPECT_EQ(no_columns.rows, 5);
            EXPECT_EQ(no_columns.cols, 0);
            EXPECT_EQ(no_columns.nnz(), 0);
            EXPECT_EQ(read_tensor<int64_t>(no_columns.row_offsets), std::vector<int64_t>(6, 1));
        }

        TYPED_TEST(Unit_SparseBuilder, ASparseVectorHoldsItsPositionsAndValues)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            const std::vector<int64_t> positions = {6, 1, 4};
            const Wide values = {{10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}};

            for (const IndexBase base : K_BOTH_BASES)
            {
                const SparseVector x = this->build_vector(positions, values, base);

                EXPECT_EQ(x.size, K_VECTOR_SIZE);
                EXPECT_EQ(x.index_base, base);
                EXPECT_EQ(x.indices.rank, 1);
                EXPECT_EQ(x.values.rank, 1);
                ASSERT_EQ(x.nnz(), 3);
                ASSERT_EQ(x.index_dtype(), sushiblas_dtype<Index>());
                ASSERT_EQ(x.dtype(), sushiblas_dtype<T>());

                EXPECT_EQ(read_tensor<Index>(x.indices), stored_indices<Index>(positions, base));
                const std::vector<T> stored = read_tensor<T>(x.values);
                for (std::size_t k = 0; k < stored.size(); ++k)
                    EXPECT_EQ(widened(stored[k]), values[k]) << "entry " << k;

                const Kernels::IndexList<Index> list{x.nnz(), x.size, Core::first_index(base),
                                                     x.indices.data_as<Index>()};
                EXPECT_EQ(find_defect(list), StructureDefect::None);
            }
        }

        TYPED_TEST(Unit_SparseBuilder, VectorArraysOfDifferentLengthsThrow)
        {
            const Wide values = {{10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}};

            EXPECT_THROW((void)this->build_vector({6, 1}, values, IndexBase::Zero),
                         std::runtime_error);
        }

        TYPED_TEST(Unit_SparseBuilder, ARepeatedVectorPositionThrows)
        {
            const Wide values = {{10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}};

            EXPECT_THROW((void)this->build_vector({6, 1, 6}, values, IndexBase::Zero),
                         std::runtime_error);
        }

        /**
         * @brief Holds the builder cases that do not depend on the element or the index type.
         */
        class Unit_SparseBuilderRules : public SushiBLASTest
        {
            protected:
                /** @brief Returns the FLOAT32 sample with INT32 indices from a builder. */
                [[nodiscard]] static SparseMatrix sample_from(const SparseBuilder& builder,
                                                              IndexBase base)
                {
                    const HostTriplets<float> host = sample_matrix<float>();
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<int32_t>(host.row_indices, base),
                        stored_indices<int32_t>(host.column_indices, base), host.values, base);
                }
        };

        TEST_F(Unit_SparseBuilderRules, ADenseTensorLargerThanOneTransferKeepsEveryEntry)
        {
            std::vector<float> host(static_cast<std::size_t>(K_LARGE_ROWS * K_LARGE_COLS), 0.0f);
            std::vector<int32_t> offsets = {0};
            std::vector<int32_t> columns;
            for (int64_t i = 0; i < K_LARGE_ROWS; ++i)
            {
                for (int64_t j = 0; j < K_LARGE_COLS; ++j)
                {
                    if ((i + K_LARGE_WEIGHT * j) % K_LARGE_MODULUS != 0)
                        continue;

                    host[static_cast<std::size_t>(i * K_LARGE_COLS + j)] = 1.0f;
                    columns.push_back(static_cast<int32_t>(j));
                }
                offsets.push_back(static_cast<int32_t>(columns.size()));
            }

            const Tensor dense =
                create_initialized_tensor<float>({K_LARGE_ROWS, K_LARGE_COLS}, host);
            const SparseBuilder builder(engine->tensor_factory());
            const SparseMatrix A = builder.matrix_from_dense(dense, DataType::Int32);

            ASSERT_EQ(A.nnz(), static_cast<int64_t>(columns.size()));
            EXPECT_EQ(read_tensor<int32_t>(A.row_offsets), offsets);
            EXPECT_EQ(read_tensor<int32_t>(A.column_indices), columns);

            const std::vector<float> values = read_tensor<float>(A.values);
            EXPECT_EQ(std::count(values.begin(), values.end(), 1.0f),
                      static_cast<std::ptrdiff_t>(values.size()));
        }

        TEST_F(Unit_SparseBuilderRules, ADenseTensorOfRankOneThrows)
        {
            const SparseBuilder builder(engine->tensor_factory());
            const Tensor dense = engine->create_tensor({5});

            EXPECT_THROW((void)builder.matrix_from_dense(dense), std::runtime_error);
        }

        TEST_F(Unit_SparseBuilderRules, ADenseTensorOfIntegersThrows)
        {
            const SparseBuilder builder(engine->tensor_factory());
            const Tensor dense = create_initialized_tensor<int32_t>({2, 2}, {1, 0, 0, 2});

            EXPECT_THROW((void)builder.matrix_from_dense(dense), std::runtime_error);
        }

        TEST_F(Unit_SparseBuilderRules, AFloatingPointIndexDtypeThrows)
        {
            const SparseBuilder builder(engine->tensor_factory());
            const Tensor dense = create_initialized_tensor<float>({2, 2}, {1.0f, 0.0f, 0.0f, 2.0f});

            EXPECT_THROW((void)builder.matrix_from_dense(dense, DataType::Float32),
                         std::runtime_error);
        }

        TEST_F(Unit_SparseBuilderRules, AStridedDenseViewThrows)
        {
            const SparseBuilder builder(engine->tensor_factory());
            const Tensor parent =
                create_initialized_tensor<float>({4, 4}, std::vector<float>(16, 1.0f));

            const int64_t dims[2] = {2, 2};
            std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
            strides[0] = 8;
            strides[1] = 2;
            const Tensor view(parent.storage, SushiRuntime::span<const int64_t>(dims, 2), strides,
                              parent.dtype, 0, parent.layout);

            EXPECT_THROW((void)builder.matrix_from_dense(view), std::runtime_error);
        }

        TEST_F(Unit_SparseBuilderRules, AnIndexBaseOutsideTheEnumerationThrows)
        {
            const SparseBuilder builder(engine->tensor_factory());
            const IndexBase outside = static_cast<IndexBase>(0);
            const Tensor dense = create_initialized_tensor<float>({2, 2}, {1.0f, 0.0f, 0.0f, 2.0f});
            const std::vector<int32_t> positions = {6, 1, 4};
            const std::vector<float> values = {10.0f, 20.0f, 30.0f};

            EXPECT_THROW((void)sample_from(builder, outside), std::runtime_error);
            EXPECT_THROW((void)builder.matrix_from_dense(dense, DataType::Int64, outside),
                         std::runtime_error);
            EXPECT_THROW((void)builder.vector_from_entries(K_VECTOR_SIZE, positions, values,
                                                           outside),
                         std::runtime_error);
        }

        TEST_F(Unit_SparseBuilderRules, TheDeviceIndexReachesTheStorageOfEveryTensor)
        {
            const std::size_t last = TestEnvironment::device_count() - 1;
            const SparseBuilder builder(engine->tensor_factory(),
                                        SushiRuntime::Residency::Shared, last);

            const SparseMatrix A = sample_from(builder, IndexBase::Zero);
            EXPECT_EQ(A.row_offsets.storage->device_index(), last);
            EXPECT_EQ(A.column_indices.storage->device_index(), last);
            EXPECT_EQ(A.values.storage->device_index(), last);

            const SparseVector x = builder.vector_from_entries(
                K_VECTOR_SIZE, std::vector<int32_t>{6, 1, 4}, std::vector<float>{1.0f, 2.0f, 3.0f});
            EXPECT_EQ(x.indices.storage->device_index(), last);
            EXPECT_EQ(x.values.storage->device_index(), last);
        }

        TEST_F(Unit_SparseBuilderRules, ADeviceIndexPastTheLastThrows)
        {
            const std::size_t past = TestEnvironment::device_count();
            const SparseBuilder builder(engine->tensor_factory(),
                                        SushiRuntime::Residency::Shared, past);

            EXPECT_THROW((void)sample_from(builder, IndexBase::Zero), std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
