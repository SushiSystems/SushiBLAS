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
 * @file test_sparse_builder.cpp
 * @brief Tests the sparse vectors and CSR matrices SparseBuilder makes from host data.
 * @author Mustafa Garip
 */

#include <array>
#include <optional>
#include <utility>
#include <initializer_list>
#include <cmath>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/sparse_storage/sparse_builder.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
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

            /** @brief A host vector of expectations or of operand values. */
            using Wide = std::vector<std::complex<double>>;

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

            /** @brief Names an element type, an index type and the index base a suite runs with. */
            template<typename Case, IndexBase Base>
            struct BasedCase : Case
            {
                /** @brief The index base the suite builds with. */
                static constexpr IndexBase K_BASE = Base;
            };

            /** @brief Lists each case of a type list once per index base. */
            template<typename Cases>
            struct WithBases;

            template<typename... Cases>
            struct WithBases<::testing::Types<Cases...>>
            {
                /** @brief The cases, zero-based first and one-based after. */
                using type = ::testing::Types<BasedCase<Cases, IndexBase::Zero>...,
                                              BasedCase<Cases, IndexBase::One>...>;
            };

            /** @brief Expects stored values to equal the expectations, entry by entry. */
            template<typename T>
            void expect_values(const std::vector<T>& actual, const Wide& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t k = 0; k < actual.size(); ++k)
                    EXPECT_EQ(widened(actual[k]), expected[k]) << "entry " << k;
            }

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
                expect_values<T>(read_tensor<T>(A.values), values);
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

            /** @brief Returns the sample entries given twice, each time as half of its value. */
            template<typename T>
            [[nodiscard]] HostTriplets<T> sample_as_halves()
            {
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
                return halves;
            }

            /** @brief Returns the zero-based positions of the vector the vector tests build. */
            [[nodiscard]] std::vector<int64_t> vector_positions()
            {
                return {6, 1, 4};
            }

            /** @brief Returns the values at those positions. */
            [[nodiscard]] Wide vector_values()
            {
                return {{10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}};
            }

            /** @brief Fills a factory slot with a factory that allocates host memory. */
            struct HostFactoryMaker
            {
                /** @brief Makes a host factory of layout in slot. */
                static void make(std::optional<TensorFactory>& slot, Layout layout)
                {
                    slot.emplace(TensorFactory::Host{}, layout);
                }
            };

            /** @brief Fills a factory slot with a factory that allocates on the shared runtime. */
            struct RuntimeFactoryMaker
            {
                /** @brief Makes a runtime factory of layout in slot. */
                static void make(std::optional<TensorFactory>& slot, Layout layout)
                {
                    slot.emplace(TestEnvironment::runtime(), layout);
                }
            };
        } // namespace

        /**
         * @brief Gives a test a tensor factory of the maker's kind, in a layout it can change.
         */
        template<typename Maker>
        class FactoryFixture : public ::testing::Test
        {
            public:
                /** @brief Makes the factory row-major. */
                FactoryFixture()
                {
                    use_layout(Layout::RowMajor);
                }

            protected:
                /** @brief Makes the factory create tensors in layout. */
                void use_layout(Layout layout)
                {
                    Maker::make(factory_, layout);
                }

                /** @brief Returns the factory the builders of the test use. */
                [[nodiscard]] TensorFactory& tensor_factory() noexcept
                {
                    return *factory_;
                }

                /** @brief Returns a tensor of T's dtype that holds a copy of data. */
                template<typename T>
                [[nodiscard]] Tensor create_initialized_tensor(std::initializer_list<int64_t> dims,
                                                               const std::vector<T>& data)
                {
                    const std::vector<int64_t> shape(dims);
                    Tensor tensor = factory_->create(
                        SushiRuntime::span<const int64_t>(shape.data(), shape.size()),
                        sushiblas_dtype<T>());
                    tensor.storage->copy_from_host(data.data(), 0, data.size() * sizeof(T));
                    return tensor;
                }

            private:
                /// @brief The factory, made by the maker.
                std::optional<TensorFactory> factory_;
        };

        /**
         * @brief Builds sparse operands and dense tensors of one element and one index type on
         *        host memory.
         */
        template<typename TypeParam>
        class SparseBuilderFixture : public FactoryFixture<HostFactoryMaker>
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief Returns the CSR matrix of host triplets, stored with base. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<T>& host, IndexBase base)
                {
                    const SparseBuilder builder(tensor_factory());
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<Index>(host.row_indices, base),
                        stored_indices<Index>(host.column_indices, base), host.values, base);
                }

                /** @brief Returns the sparse vector of zero-based positions, stored with base. */
                [[nodiscard]] SparseVector build_vector(const std::vector<int64_t>& positions,
                                                        const Wide& values, IndexBase base)
                {
                    const SparseBuilder builder(tensor_factory());
                    return builder.vector_from_entries(K_VECTOR_SIZE,
                                                       stored_indices<Index>(positions, base),
                                                       narrowed_values(values), base);
                }

                /** @brief Returns the CSR matrix of a dense tensor, with indices of index_dtype. */
                [[nodiscard]] SparseMatrix compress(const Tensor& dense, DataType index_dtype,
                                                    IndexBase base)
                {
                    const SparseBuilder builder(tensor_factory());
                    return builder.matrix_from_dense(dense, index_dtype, base);
                }

                /**
                 * @brief Returns a dense tensor of the factory's layout holding a matrix.
                 * @param by_rows rows * cols elements, row by row; neither extent is 0.
                 */
                [[nodiscard]] Tensor dense_tensor(int64_t rows, int64_t cols, const Wide& by_rows)
                {
                    const bool row_major = tensor_factory().layout() == Layout::RowMajor;
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

                /** @brief Returns host expectations narrowed to T. */
                [[nodiscard]] static std::vector<T> narrowed_values(const Wide& values)
                {
                    std::vector<T> host(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        host[i] = narrowed<T>(values[i]);
                    return host;
                }

                /** @brief Returns the 2 x 2 matrix of by_rows compressed with 64-bit indices. */
                [[nodiscard]] SparseMatrix compress_two_by_two(const Wide& by_rows)
                {
                    return compress(dense_tensor(2, 2, by_rows), DataType::Int64, IndexBase::Zero);
                }
        };

        /** @brief Builds the sparse operands of every case, with zero-based indices. */
        template<typename TypeParam>
        class Small_SparseBuilder : public SparseBuilderFixture<TypeParam>
        {
        };

        /** @brief Builds the sparse operands of every case under both index bases. */
        template<typename TypeParam>
        class Small_SparseBuilderBased : public SparseBuilderFixture<TypeParam>
        {
        };

        SB_TYPED_TEST_SUITE(Small_SparseBuilder, SparseCases);
        SB_TYPED_TEST_SUITE(Small_SparseBuilderBased, WithBases<SparseCases>::type);

        TYPED_TEST(Small_SparseBuilderBased, TheSampleTripletsGiveTheSampleMatrix)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            expect_sample<T, Index>(this->build(sample_matrix<T>(), TypeParam::K_BASE),
                                    TypeParam::K_BASE);
        }

        TYPED_TEST(Small_SparseBuilderBased, EntriesGivenTwiceAsHalvesGiveTheSameSevenEntries)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            expect_sample<T, Index>(this->build(sample_as_halves<T>(), TypeParam::K_BASE),
                                    TypeParam::K_BASE);
        }

        TYPED_TEST(Small_SparseBuilderBased, AMatrixWithoutEntriesHasNoStorageForThem)
        {
            using T = typename TypeParam::Value;

            HostTriplets<T> host;
            host.rows = 3;
            host.cols = 4;

            const SparseMatrix A = this->build(host, TypeParam::K_BASE);

            EXPECT_EQ(A.nnz(), 0);
            EXPECT_TRUE(A.values.storage == nullptr);
            EXPECT_TRUE(A.column_indices.storage == nullptr);
        }

        TYPED_TEST(Small_SparseBuilderBased, AMatrixWithoutEntriesHasRowOffsetsAtTheFirstIndex)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            HostTriplets<T> host;
            host.rows = 3;
            host.cols = 4;

            const SparseMatrix A = this->build(host, TypeParam::K_BASE);

            EXPECT_EQ(read_tensor<Index>(A.row_offsets),
                      std::vector<Index>(4, static_cast<Index>(Core::first_index(TypeParam::K_BASE))));
        }

        TYPED_TEST(Small_SparseBuilderBased, ADenseTensorCompressesToTheSample)
        {
            using T = typename TypeParam::Value;

            const Tensor dense = this->dense_tensor(4, 5, sample_by_rows<T>());

            expect_sample<T, int64_t>(this->compress(dense, DataType::Int64, TypeParam::K_BASE),
                                      TypeParam::K_BASE);
        }

        TYPED_TEST(Small_SparseBuilderBased, ASparseVectorHoldsItsPositions)
        {
            using Index = typename TypeParam::IndexType;

            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), TypeParam::K_BASE);

            EXPECT_EQ(read_tensor<Index>(x.indices),
                      stored_indices<Index>(vector_positions(), TypeParam::K_BASE));
        }

        TYPED_TEST(Small_SparseBuilderBased, ASparseVectorHoldsItsValues)
        {
            using T = typename TypeParam::Value;

            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), TypeParam::K_BASE);

            expect_values<T>(read_tensor<T>(x.values), vector_values());
        }

        TYPED_TEST(Small_SparseBuilderBased, ASparseVectorKeepsItsIndexBase)
        {
            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), TypeParam::K_BASE);

            EXPECT_EQ(x.index_base, TypeParam::K_BASE);
        }

        TYPED_TEST(Small_SparseBuilder, TheThreeTensorsHaveRankOne)
        {
            using T = typename TypeParam::Value;

            const SparseMatrix A = this->build(sample_matrix<T>(), IndexBase::Zero);

            EXPECT_EQ(A.row_offsets.rank, 1);
            EXPECT_EQ(A.column_indices.rank, 1);
            EXPECT_EQ(A.values.rank, 1);
        }

        TYPED_TEST(Small_SparseBuilder, TheThreeTensorsHoldOneElementPerRowPlusOneAndPerEntry)
        {
            using T = typename TypeParam::Value;

            const SparseMatrix A = this->build(sample_matrix<T>(), IndexBase::Zero);

            EXPECT_EQ(A.row_offsets.num_elements, 5);
            EXPECT_EQ(A.column_indices.num_elements, 7);
            EXPECT_EQ(A.values.num_elements, 7);
        }

        TYPED_TEST(Small_SparseBuilder, TheOffsetsAndColumnsHaveTheDtypeOfTheHostIndex)
        {
            using T = typename TypeParam::Value;
            using Index = typename TypeParam::IndexType;

            const SparseMatrix A = this->build(sample_matrix<T>(), IndexBase::Zero);

            EXPECT_EQ(A.row_offsets.dtype, sushiblas_dtype<Index>());
            EXPECT_EQ(A.column_indices.dtype, sushiblas_dtype<Index>());
        }

        TYPED_TEST(Small_SparseBuilder, TheValuesHaveTheDtypeOfTheHostValue)
        {
            using T = typename TypeParam::Value;

            const SparseMatrix A = this->build(sample_matrix<T>(), IndexBase::Zero);

            EXPECT_EQ(A.values.dtype, sushiblas_dtype<T>());
        }

        TYPED_TEST(Small_SparseBuilder, ADenseTensorCompressesToTheSampleFromAColumnMajorTensor)
        {
            using T = typename TypeParam::Value;
            this->use_layout(Layout::ColumnMajor);
            const Tensor dense = this->dense_tensor(4, 5, sample_by_rows<T>());

            expect_sample<T, int64_t>(this->compress(dense, DataType::Int64, IndexBase::Zero),
                                      IndexBase::Zero);
        }

        TYPED_TEST(Small_SparseBuilder, ADenseTensorCompressesToTheSampleWithInt32Indices)
        {
            using T = typename TypeParam::Value;

            const Tensor dense = this->dense_tensor(4, 5, sample_by_rows<T>());

            expect_sample<T, int32_t>(this->compress(dense, DataType::Int32, IndexBase::Zero),
                                      IndexBase::Zero);
        }

        TYPED_TEST(Small_SparseBuilder, ADenseTensorDropsANegativeZero)
        {
            using T = typename TypeParam::Value;
            const std::complex<double> negative_zero(-0.0, -0.0);

            const SparseMatrix A =
                this->compress_two_by_two({negative_zero, {0.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});

            EXPECT_EQ(A.nnz(), 1);
            expect_values<T>(read_tensor<T>(A.values), {{3.0, 0.0}});
        }

        TYPED_TEST(Small_SparseBuilder, ADenseTensorKeepsANaN)
        {
            using T = typename TypeParam::Value;
            const std::complex<double> nan(std::nan(""), 0.0);

            const SparseMatrix A =
                this->compress_two_by_two({{0.0, 0.0}, nan, {0.0, 0.0}, {3.0, 0.0}});

            ASSERT_EQ(A.nnz(), 2);
            EXPECT_TRUE(std::isnan(widened(read_tensor<T>(A.values)[0]).real()));
        }

        TYPED_TEST(Small_SparseBuilder, VectorArraysOfDifferentLengthsAreRefused)
        {
            EXPECT_THROW((void)this->build_vector({6, 1}, vector_values(), IndexBase::Zero),
                         std::runtime_error);
        }

        TYPED_TEST(Small_SparseBuilder, ARepeatedVectorPositionIsRefused)
        {
            EXPECT_THROW((void)this->build_vector({6, 1, 6}, vector_values(), IndexBase::Zero),
                         std::runtime_error);
        }

        TYPED_TEST(Small_SparseBuilder, ASparseVectorKeepsItsSize)
        {
            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), IndexBase::Zero);

            EXPECT_EQ(x.size, K_VECTOR_SIZE);
        }

        TYPED_TEST(Small_SparseBuilder, ASparseVectorCountsItsEntries)
        {
            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), IndexBase::Zero);

            EXPECT_EQ(x.nnz(), 3);
        }

        TYPED_TEST(Small_SparseBuilder, TheVectorArraysHaveRankOne)
        {
            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), IndexBase::Zero);

            EXPECT_EQ(x.indices.rank, 1);
            EXPECT_EQ(x.values.rank, 1);
        }

        TYPED_TEST(Small_SparseBuilder, TheVectorIndicesHaveTheDtypeOfTheHostIndex)
        {
            using Index = typename TypeParam::IndexType;

            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), IndexBase::Zero);

            EXPECT_EQ(x.index_dtype(), sushiblas_dtype<Index>());
        }

        TYPED_TEST(Small_SparseBuilder, TheVectorValuesHaveTheDtypeOfTheHostValue)
        {
            using T = typename TypeParam::Value;

            const SparseVector x =
                this->build_vector(vector_positions(), vector_values(), IndexBase::Zero);

            EXPECT_EQ(x.dtype(), sushiblas_dtype<T>());
        }

        /** @brief The extents of a dense tensor that holds no element. */
        struct EmptyDenseRow
        {
            const char* name;   /**< The row suffix. */
            int64_t rows;       /**< Rows of the dense tensor. */
            int64_t cols;       /**< Columns of the dense tensor. */
        };

        /** @brief Names a row, which gtest takes as the test suffix. */
        static std::string empty_dense_row_name(
            const ::testing::TestParamInfo<EmptyDenseRow>& info)
        {
            return info.param.name;
        }

        /** @brief Compresses a FLOAT32 dense tensor with one zero extent at a time. */
        class Small_SparseBuilderEmptyDense : public FactoryFixture<HostFactoryMaker>,
                                              public ::testing::WithParamInterface<EmptyDenseRow>
        {
            protected:
                /** @brief Returns the one-based matrix a dense tensor of the row's extents gives. */
                [[nodiscard]] SparseMatrix compress_the_row()
                {
                    const int64_t dims[2] = {GetParam().rows, GetParam().cols};
                    const Tensor dense = tensor_factory().create(
                        SushiRuntime::span<const int64_t>(dims, 2), DataType::Float32);
                    const SparseBuilder builder(tensor_factory());
                    return builder.matrix_from_dense(dense, DataType::Int64, IndexBase::One);
                }
        };

        TEST_P(Small_SparseBuilderEmptyDense, KeepsTheExtentsOfTheDenseTensor)
        {
            const SparseMatrix A = compress_the_row();

            EXPECT_EQ(A.rows, GetParam().rows);
            EXPECT_EQ(A.cols, GetParam().cols);
        }

        TEST_P(Small_SparseBuilderEmptyDense, HasNoEntries)
        {
            EXPECT_EQ(compress_the_row().nnz(), 0);
        }

        TEST_P(Small_SparseBuilderEmptyDense, HasRowOffsetsAtTheFirstIndex)
        {
            const SparseMatrix A = compress_the_row();

            EXPECT_EQ(read_tensor<int64_t>(A.row_offsets),
                      std::vector<int64_t>(static_cast<std::size_t>(GetParam().rows) + 1, 1));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SparseBuilderEmptyDense,
                                 ::testing::Values(EmptyDenseRow{"NoRows", 0, 5},
                                                   EmptyDenseRow{"NoColumns", 5, 0}),
                                 empty_dense_row_name);

        /**
         * @brief Holds the builder cases that do not depend on the element or the index type.
         */
        class Small_SparseBuilderRules : public FactoryFixture<HostFactoryMaker>
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

                /** @brief Returns the 2 x 2 FLOAT32 tensor of 1, 0, 0, 2. */
                [[nodiscard]] Tensor small_dense()
                {
                    return create_initialized_tensor<float>({2, 2}, {1.0f, 0.0f, 0.0f, 2.0f});
                }
        };

        TEST_F(Small_SparseBuilderRules, ADenseTensorOfRankOneIsRefused)
        {
            const SparseBuilder builder(tensor_factory());
            const int64_t dims[1] = {5};
            const Tensor dense =
                tensor_factory().create(SushiRuntime::span<const int64_t>(dims, 1));

            EXPECT_THROW((void)builder.matrix_from_dense(dense), std::runtime_error);
        }

        TEST_F(Small_SparseBuilderRules, ADenseTensorOfIntegersIsRefused)
        {
            const SparseBuilder builder(tensor_factory());
            const Tensor dense = create_initialized_tensor<int32_t>({2, 2}, {1, 0, 0, 2});

            EXPECT_THROW((void)builder.matrix_from_dense(dense), std::runtime_error);
        }

        TEST_F(Small_SparseBuilderRules, AFloatingPointIndexDtypeIsRefused)
        {
            const SparseBuilder builder(tensor_factory());

            EXPECT_THROW((void)builder.matrix_from_dense(small_dense(), DataType::Float32),
                         std::runtime_error);
        }

        TEST_F(Small_SparseBuilderRules, AStridedDenseViewIsRefused)
        {
            const SparseBuilder builder(tensor_factory());
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

        TEST_F(Small_SparseBuilderRules, TripletsRefuseAnIndexBaseOutsideTheEnumeration)
        {
            const SparseBuilder builder(tensor_factory());

            EXPECT_THROW((void)sample_from(builder, static_cast<IndexBase>(0)),
                         std::runtime_error);
        }

        TEST_F(Small_SparseBuilderRules, ADenseTensorRefusesAnIndexBaseOutsideTheEnumeration)
        {
            const SparseBuilder builder(tensor_factory());

            EXPECT_THROW((void)builder.matrix_from_dense(small_dense(), DataType::Int64,
                                                         static_cast<IndexBase>(0)),
                         std::runtime_error);
        }

        TEST_F(Small_SparseBuilderRules, VectorEntriesRefuseAnIndexBaseOutsideTheEnumeration)
        {
            const SparseBuilder builder(tensor_factory());

            EXPECT_THROW((void)builder.vector_from_entries(K_VECTOR_SIZE,
                                                           std::vector<int32_t>{6, 1, 4},
                                                           std::vector<float>{10.0f, 20.0f, 30.0f},
                                                           static_cast<IndexBase>(0)),
                         std::runtime_error);
        }

        TEST_F(Small_SparseBuilderRules, ADeviceIndexPastTheLastIsRefused)
        {
            const SparseBuilder builder(tensor_factory(), SushiRuntime::Residency::Shared, 1);

            EXPECT_THROW((void)sample_from(builder, IndexBase::Zero), std::runtime_error);
        }

        /**
         * @brief Holds the builder cases that reach a device through the shared runtime.
         */
        class Medium_SparseBuilderDevice : public FactoryFixture<RuntimeFactoryMaker>
        {
            protected:
                /** @brief Holds a FLOAT32 matrix with a sparse pattern and its expected CSR arrays. */
                struct LargeDense
                {
                    std::vector<float> elements;        /**< Row by row, K_LARGE_ROWS by K_LARGE_COLS. */
                    std::vector<int32_t> offsets;       /**< The expected row offsets. */
                    std::vector<int32_t> columns;       /**< The expected column indices. */
                };

                /** @brief Returns the matrix of K_LARGE_MODULUS and its CSR arrays. */
                [[nodiscard]] static LargeDense large_dense()
                {
                    LargeDense large;
                    large.elements.assign(static_cast<std::size_t>(K_LARGE_ROWS * K_LARGE_COLS),
                                          0.0f);
                    large.offsets = {0};
                    for (int64_t i = 0; i < K_LARGE_ROWS; ++i)
                    {
                        for (int64_t j = 0; j < K_LARGE_COLS; ++j)
                        {
                            if ((i + K_LARGE_WEIGHT * j) % K_LARGE_MODULUS != 0)
                                continue;

                            large.elements[static_cast<std::size_t>(i * K_LARGE_COLS + j)] = 1.0f;
                            large.columns.push_back(static_cast<int32_t>(j));
                        }
                        large.offsets.push_back(static_cast<int32_t>(large.columns.size()));
                    }
                    return large;
                }
        };

        TEST_F(Medium_SparseBuilderDevice, ADenseTensorLargerThanOneTransferKeepsEveryEntry)
        {
            const LargeDense large = large_dense();
            const Tensor dense =
                create_initialized_tensor<float>({K_LARGE_ROWS, K_LARGE_COLS}, large.elements);
            const SparseBuilder builder(tensor_factory());

            const SparseMatrix A = builder.matrix_from_dense(dense, DataType::Int32);

            ASSERT_EQ(A.nnz(), static_cast<int64_t>(large.columns.size()));
            EXPECT_EQ(read_tensor<int32_t>(A.row_offsets), large.offsets);
            EXPECT_EQ(read_tensor<int32_t>(A.column_indices), large.columns);
        }

        TEST_F(Medium_SparseBuilderDevice, TheDeviceIndexReachesTheStorageOfEveryMatrixTensor)
        {
            const std::size_t last = TestEnvironment::device_count() - 1;
            const HostTriplets<float> host = sample_matrix<float>();
            const SparseBuilder builder(tensor_factory(), SushiRuntime::Residency::Shared, last);

            const SparseMatrix A = builder.matrix_from_triplets(
                host.rows, host.cols, stored_indices<int32_t>(host.row_indices, IndexBase::Zero),
                stored_indices<int32_t>(host.column_indices, IndexBase::Zero), host.values);

            EXPECT_EQ(A.row_offsets.storage->device_index(), last);
            EXPECT_EQ(A.column_indices.storage->device_index(), last);
            EXPECT_EQ(A.values.storage->device_index(), last);
        }

        TEST_F(Medium_SparseBuilderDevice, TheDeviceIndexReachesTheStorageOfEveryVectorTensor)
        {
            const std::size_t last = TestEnvironment::device_count() - 1;
            const SparseBuilder builder(tensor_factory(), SushiRuntime::Residency::Shared, last);

            const SparseVector x = builder.vector_from_entries(
                K_VECTOR_SIZE, std::vector<int32_t>{6, 1, 4}, std::vector<float>{1.0f, 2.0f, 3.0f});

            EXPECT_EQ(x.indices.storage->device_index(), last);
            EXPECT_EQ(x.values.storage->device_index(), last);
        }
    } // namespace Test
} // namespace SushiBLAS
