/**************************************************************************/
/* test_sparse_spmv.cpp                                                   */
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
 * @file test_sparse_spmv.cpp
 * @brief Tests SparseOps::spmv against host products, and the transposed SYCL kernel on a matrix
 *        whose heaviest row spans several row blocks.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <limits>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/sparse_kernels/sparse.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include "support/forced_mode_checks.hpp"
#include "support/sparse_conj_case.hpp"
#include "support/sparse_test_sweeps.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance: 128 FLOAT32 roundings of a value of order one,
        /// for an element of the 4 x 5 sample.
        constexpr double K_SAMPLE_TOLERANCE = absolute_tolerance<float>(128);

        /// @brief Rows of the integer-valued matrix; above one work-group on every device.
        constexpr int64_t K_EXACT_ROWS = 1000;

        /// @brief Columns of the integer-valued matrix; not equal to its rows.
        constexpr int64_t K_EXACT_COLS = 777;

        /// @brief Entries drawn for each row of the integer-valued matrix.
        constexpr int64_t K_EXACT_PER_ROW = 8;

        /// @brief Seed of the integer-valued matrix.
        constexpr std::uint32_t K_EXACT_SEED = 20261007u;

        /// @brief The value the elements between those of a strided view hold.
        constexpr double K_FILLER = -7.0;

        /// @brief Empty rows before the first stored entry of the block-partition matrix.
        constexpr int64_t K_LEADING_EMPTY_ROWS = 3;

        /// @brief Rows of the block-partition matrix that store entries.
        constexpr int64_t K_STORED_ROWS = 100;

        /// @brief The stored row of the block-partition matrix that holds K_HEAVY_ENTRIES.
        constexpr int64_t K_HEAVY_ROW = 50;

        /// @brief Entries of the heavy row; every other stored row holds one.
        constexpr int64_t K_HEAVY_ENTRIES = 900;

        /// @brief Empty rows after the last stored entry of the block-partition matrix.
        constexpr int64_t K_TRAILING_EMPTY_ROWS = 2;

        namespace
        {
            /** @brief A host vector of expectations or of operand values. */
            using Wide = std::vector<std::complex<double>>;

            /** @brief Returns n elements cycling through -2, -1, 0, 1, 2. */
            [[nodiscard]] Wide cycling_vector(std::size_t n)
            {
                Wide cycled(n);
                for (std::size_t i = 0; i < n; ++i)
                    cycled[i] = std::complex<double>(static_cast<double>(i % 5) - 2.0, 0.0);
                return cycled;
            }

            /** @brief Returns the real parts of a vector as floats. */
            [[nodiscard]] std::vector<float> floats_of(const Wide& values)
            {
                std::vector<float> narrowed_values(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    narrowed_values[i] = static_cast<float>(values[i].real());
                return narrowed_values;
            }

            /**
             * @brief Returns the matrix whose rows split into row blocks unevenly: three empty
             *        rows, a hundred stored rows of which one holds 900 entries and the others
             *        one, and two empty rows. Every entry is 1 in column 0.
             */
            [[nodiscard]] HostTriplets<float> block_partition_matrix()
            {
                HostTriplets<float> m;
                m.rows = K_LEADING_EMPTY_ROWS + K_STORED_ROWS + K_TRAILING_EMPTY_ROWS;
                m.cols = 1;
                for (int64_t row = 0; row < K_STORED_ROWS; ++row)
                {
                    const int64_t entries = row == K_HEAVY_ROW ? K_HEAVY_ENTRIES : 1;
                    for (int64_t entry = 0; entry < entries; ++entry)
                    {
                        m.row_indices.push_back(K_LEADING_EMPTY_ROWS + row);
                        m.column_indices.push_back(0);
                        m.values.push_back(1.0f);
                    }
                }
                return m;
            }

            /** @brief Holds a count of row blocks the kernel is launched with. */
            struct BlockRow
            {
                const char* name;   /**< The row suffix. */
                std::size_t blocks; /**< The row blocks of the scatter. */
            };

            /** @brief Returns one block per entry range of the heavy row, a few, and the maximum. */
            std::vector<BlockRow> block_rows()
            {
                return {{"OneBlock", 1}, {"TwoBlocks", 2}, {"SevenBlocks", 7},
                        {"TheMaximumOfBlocks", Kernels::SYCL::K_MAX_SCATTER_BLOCKS}};
            }

            /// @brief Names a row, which gtest takes as the test suffix.
            std::string block_row_name(const ::testing::TestParamInfo<BlockRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /**
         * @brief Builds CSR matrices from host triplets and runs spmv in a chosen execution mode.
         */
        template<typename TypeParam>
        class Medium_SparseSpmvTest : public SushiBLASTest
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief A host vector of expectations or of operand values. */
                using Wide = std::vector<std::complex<double>>;

                /** @brief Skips the HALF suite on a device without the fp16 aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (std::is_same_v<T, sycl::half> &&
                        !TestEnvironment::queue().get_device().has(
                            sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }

                /** @brief Returns the CSR matrix of host triplets, stored with base. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<T>& host, Core::IndexBase base)
                {
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<Index>(host.row_indices, base),
                        stored_indices<Index>(host.column_indices, base), host.values, base);
                }

                /** @brief Returns a rank-1 tensor holding values narrowed to T. */
                [[nodiscard]] Tensor vector_of(const Wide& values)
                {
                    std::vector<T> host(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        host[i] = narrowed<T>(values[i]);

                    const int64_t count = static_cast<int64_t>(values.size());
                    return create_initialized_tensor<T>({count}, host);
                }

                /** @brief Returns a rank-1 tensor of T that holds no element. */
                [[nodiscard]] Tensor vector_without_elements()
                {
                    return engine->create_tensor({0}, sushiblas_dtype<T>());
                }

                /**
                 * @brief Returns a CSR matrix whose three arrays the caller gives as they are.
                 * @param offsets Zero-based; rows + 1 of them.
                 * @param columns Zero-based; a (row, column) pair may be given twice.
                 */
                [[nodiscard]] SparseMatrix matrix_of_arrays(int64_t rows, int64_t cols,
                                                            const std::vector<int64_t>& offsets,
                                                            const std::vector<int64_t>& columns,
                                                            const Wide& values,
                                                            Core::IndexBase base)
                {
                    SparseMatrix A;
                    A.rows = rows;
                    A.cols = cols;
                    A.index_base = base;
                    A.row_offsets = create_initialized_tensor<Index>(
                        {static_cast<int64_t>(offsets.size())},
                        stored_indices<Index>(offsets, base));
                    A.column_indices = create_initialized_tensor<Index>(
                        {static_cast<int64_t>(columns.size())},
                        stored_indices<Index>(columns, base));
                    A.values = vector_of(values);
                    return A;
                }

                /** @brief Returns a tensor of twice the length: values at its even elements. */
                [[nodiscard]] Tensor interleaved(const Wide& values)
                {
                    Wide spread(2 * values.size(), std::complex<double>(K_FILLER, 0.0));
                    for (std::size_t i = 0; i < values.size(); ++i)
                        spread[2 * i] = values[i];
                    return vector_of(spread);
                }

                /** @brief Returns the stride-2 view of the even elements of a tensor. */
                [[nodiscard]] static Tensor every_second(const Tensor& wide)
                {
                    const int64_t dims[1] = {wide.num_elements / 2};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(wide.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, wide.dtype, 0, wide.layout);
                }

                /** @brief Returns y after y = alpha * op(A) * x + beta * y recorded in mode. */
                [[nodiscard]] std::vector<T> run(ExecutionMode mode, const HostTriplets<T>& host,
                                                 Core::IndexBase base, Core::Transpose trans,
                                                 std::complex<double> alpha, const Wide& x,
                                                 std::complex<double> beta, const Wide& y)
                {
                    set_mode(mode);
                    const SparseMatrix A = build(host, base);
                    const Tensor tx = vector_of(x);
                    Tensor ty = vector_of(y);
                    SushiBLAS::SparseOps(recorder()).spmv(A, tx, ty, trans, BLASScalar(alpha),
                                                       BLASScalar(beta));
                    recording->run().value();
                    return read_tensor<T>(ty);
                }

                /** @brief Expects every element within tolerance of its host expectation. */
                static void expect_near(const std::vector<T>& actual, const Wide& expected,
                                        double tolerance)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const std::complex<double> got = widened(actual[i]);
                        EXPECT_NEAR(got.real(), expected[i].real(), tolerance) << "element " << i;
                        EXPECT_NEAR(got.imag(), expected[i].imag(), tolerance) << "element " << i;
                    }
                }

                /** @brief Expects every element to equal its host expectation exactly. */
                static void expect_equal(const std::vector<T>& actual, const Wide& expected)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const std::complex<double> got = widened(actual[i]);
                        EXPECT_EQ(got.real(), expected[i].real()) << "element " << i;
                        EXPECT_EQ(got.imag(), expected[i].imag()) << "element " << i;
                    }
                }

                /**
                 * @brief Expects the host product of a real T to equal the figures worked out by
                 *        hand; a complex T has no such figures and passes.
                 */
                static void expect_real_product_by_hand(const Wide& expected,
                                                        const std::vector<double>& by_hand)
                {
                    if constexpr (!Core::is_complex_v<T>)
                    {
                        ASSERT_EQ(expected.size(), by_hand.size());
                        for (std::size_t i = 0; i < by_hand.size(); ++i)
                            EXPECT_EQ(expected[i].real(), by_hand[i]) << "element " << i;
                    }
                }

                /**
                 * @brief Expects the even elements near expected and the odd ones still K_FILLER.
                 * @param wide The elements of a tensor interleaved() made, after the call.
                 */
                static void expect_interleaved(const std::vector<T>& wide, const Wide& expected,
                                               double tolerance)
                {
                    ASSERT_EQ(wide.size(), 2 * expected.size());
                    std::vector<T> written(expected.size());
                    std::vector<T> skipped(expected.size());
                    for (std::size_t i = 0; i < expected.size(); ++i)
                    {
                        written[i] = wide[2 * i];
                        skipped[i] = wide[2 * i + 1];
                    }

                    expect_near(written, expected, tolerance);
                    expect_equal(skipped,
                                 Wide(expected.size(), std::complex<double>(K_FILLER, 0.0)));
                }
        };

        SB_TYPED_TEST_SUITE(Medium_SparseSpmvTest, SparseCases);

        TYPED_TEST(Medium_SparseSpmvTest, TheSampleProductMatchesTheHostInEveryModeAndBase)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide x = counting(5, 1.0);
            const Wide y(4, std::complex<double>(1.0, 0.0));
            const std::complex<double> alpha(2.0, 0.0);
            const std::complex<double> beta(-1.0, 0.0);
            const Core::Transpose trans = Core::Transpose::NoTrans;

            const Wide expected = host_product(host, trans, alpha, x, beta, y);
            // A * x is (9, 0, 43, 41) for x = (1, 2, 3, 4, 5).
            this->expect_real_product_by_hand(expected, {17.0, -1.0, 85.0, 81.0});

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                this->expect_near(this->run(mode, host, base, trans, alpha, x, beta, y), expected,
                                  K_SAMPLE_TOLERANCE);
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, TheTransposedSampleProductMatchesTheHost)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide x = counting(4, 1.0);
            const Wide y(5, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Core::Transpose trans = Core::Transpose::Trans;

            const Wide expected = host_product(host, trans, one, x, zero, y);
            this->expect_real_product_by_hand(expected, {25.0, 9.0, 12.0, 2.0, 43.0});

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                this->expect_near(this->run(mode, host, base, trans, one, x, zero, y), expected,
                                  K_SAMPLE_TOLERANCE);
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, ConjTransConjugatesAComplexMatrixAndNothingElse)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide y(5, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Core::IndexBase base = Core::IndexBase::Zero;

            // x selects row 0, whose entries are 1 + 2i in column 0 and 2 - i in column 3.
            Wide x(4, zero);
            x[0] = conj_selector<T>();

            in_each_forced_mode([&](ExecutionMode mode) {
                const std::vector<T> plain =
                    this->run(mode, host, base, Core::Transpose::Trans, one, x, zero, y);
                const std::vector<T> conjugated =
                    this->run(mode, host, base, Core::Transpose::ConjTrans, one, x, zero, y);

                this->expect_equal(plain, conj_probe_product<T>(false));
                this->expect_equal(conjugated, conj_probe_product<T>(true));
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, IntegerDataIsExactInBothModesForEveryTranspose)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = integer_valued_matrix<T>(K_EXACT_ROWS, K_EXACT_COLS,
                                                                  K_EXACT_PER_ROW, K_EXACT_SEED);
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> minus_one(-1.0, 0.0);
            const Core::IndexBase base = Core::IndexBase::Zero;

            in_each_transpose([&](Core::Transpose trans) {
                const bool transposed = Core::is_transposed(trans);
                const int64_t inputs = transposed ? host.rows : host.cols;
                const int64_t outputs = transposed ? host.cols : host.rows;

                // x cycles through -2, -1, 0, 1, 2 and y starts at 1.
                const Wide x = cycling_vector(static_cast<std::size_t>(inputs));
                const Wide y(static_cast<std::size_t>(outputs), one);

                const Wide expected = host_product(host, trans, one, x, minus_one, y);
                in_each_forced_mode([&](ExecutionMode mode) {
                    this->expect_equal(this->run(mode, host, base, trans, one, x, minus_one, y),
                                       expected);
                });
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, ABetaOfZeroDoesNotReadY)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const double nan = std::numeric_limits<double>::quiet_NaN();

            in_plain_and_transposed([&](Core::Transpose trans) {
                const bool transposed = Core::is_transposed(trans);
                const std::size_t inputs = transposed ? 4 : 5;
                const std::size_t outputs = transposed ? 5 : 4;

                const Wide x = counting(inputs, 1.0);
                const Wide poisoned(outputs, std::complex<double>(nan, nan));
                const Wide expected = host_product(host, trans, one, x, zero, Wide(outputs, zero));

                in_each_forced_mode([&](ExecutionMode mode) {
                    this->expect_near(this->run(mode, host, Core::IndexBase::Zero, trans, one, x,
                                                zero, poisoned),
                                      expected, K_SAMPLE_TOLERANCE);
                });
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, AMatrixWithoutEntriesScalesYByBeta)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            HostTriplets<T> host;
            host.rows = 3;
            host.cols = 4;
            const std::complex<double> alpha(3.0, 0.0);
            const std::complex<double> beta(-2.0, 0.0);

            in_each_transpose([&](Core::Transpose trans) {
                const bool transposed = Core::is_transposed(trans);
                const Wide x = counting(transposed ? 3 : 4, 1.0);
                const Wide y = counting(transposed ? 4 : 3, 1.0);

                const Wide expected = host_product(host, trans, alpha, x, beta, y);
                in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                    this->expect_equal(this->run(mode, host, base, trans, alpha, x, beta, y),
                                       expected);
                });
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, AMatrixWithoutRowsScalesYByBetaUnderTrans)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            HostTriplets<T> no_rows;
            no_rows.cols = 5;
            const BLASScalar alpha = 3.0;
            const BLASScalar beta = -2.0;
            const Wide scaled = {{-2.0, 0.0}, {-4.0, 0.0}, {-6.0, 0.0}, {-8.0, 0.0}, {-10.0, 0.0}};

            in_each_forced_mode([&](ExecutionMode mode) {
                this->set_mode(mode);
                const SparseMatrix wide_matrix = this->build(no_rows, Core::IndexBase::Zero);
                // The side of length 5 is y under TRANS for a matrix of 0 x 5.
                const Tensor nothing = this->vector_without_elements();
                Tensor y = this->vector_of(counting(5, 1.0));

                SparseOps(this->recorder()).spmv(wide_matrix, nothing, y, Core::Transpose::Trans,
                                                 alpha, beta);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(y), scaled);
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, AMatrixWithoutColumnsScalesYByBeta)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            HostTriplets<T> no_columns;
            no_columns.rows = 5;
            const BLASScalar alpha = 3.0;
            const BLASScalar beta = -2.0;
            const Wide scaled = {{-2.0, 0.0}, {-4.0, 0.0}, {-6.0, 0.0}, {-8.0, 0.0}, {-10.0, 0.0}};

            in_each_forced_mode([&](ExecutionMode mode) {
                this->set_mode(mode);
                const SparseMatrix tall_matrix = this->build(no_columns, Core::IndexBase::One);
                // The side of length 5 is y under NO_TRANS for a matrix of 5 x 0.
                const Tensor nothing = this->vector_without_elements();
                Tensor y = this->vector_of(counting(5, 1.0));

                SparseOps(this->recorder()).spmv(tall_matrix, nothing, y,
                                                 Core::Transpose::NoTrans, alpha, beta);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(y), scaled);
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, AMatrixWithoutRowsRecordsOneTaskForAnEmptyResult)
        {
            using T = typename TypeParam::Value;

            HostTriplets<T> no_rows;
            no_rows.cols = 5;

            in_each_forced_mode([&](ExecutionMode mode) {
                this->set_mode(mode);
                const SparseMatrix wide_matrix = this->build(no_rows, Core::IndexBase::Zero);
                const Tensor x = this->vector_of(counting(5, 1.0));
                Tensor no_result = this->vector_without_elements();

                SparseOps(this->recorder()).spmv(wide_matrix, x, no_result,
                                                 Core::Transpose::NoTrans, 3.0, -2.0);

                EXPECT_EQ(this->recording->recorded_count(), 1u);
                EXPECT_TRUE(this->recording->run_succeeds());
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, AMatrixWithoutColumnsRecordsOneTaskForAnEmptyResult)
        {
            using T = typename TypeParam::Value;

            HostTriplets<T> no_columns;
            no_columns.rows = 5;

            in_each_forced_mode([&](ExecutionMode mode) {
                this->set_mode(mode);
                const SparseMatrix tall_matrix = this->build(no_columns, Core::IndexBase::One);
                const Tensor x = this->vector_of(counting(5, 1.0));
                Tensor no_result = this->vector_without_elements();

                SparseOps(this->recorder()).spmv(tall_matrix, x, no_result,
                                                 Core::Transpose::Trans, 3.0, -2.0);

                EXPECT_EQ(this->recording->recorded_count(), 1u);
                EXPECT_TRUE(this->recording->run_succeeds());
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, StridedVectorsGiveTheSampleProduct)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            const HostTriplets<T> host = sample_matrix<T>();
            const Wide x = counting(5, 1.0);
            const Wide y(4, std::complex<double>(1.0, 0.0));
            const std::complex<double> alpha(2.0, 0.0);
            const std::complex<double> beta(-1.0, 0.0);
            const Core::Transpose trans = Core::Transpose::NoTrans;
            const Wide expected = host_product(host, trans, alpha, x, beta, y);

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                this->set_mode(mode);
                const SparseMatrix A = this->build(host, base);
                const Tensor wide_x = this->interleaved(x);
                const Tensor wide_y = this->interleaved(y);
                const Tensor tx = this->every_second(wide_x);
                Tensor ty = this->every_second(wide_y);

                SparseOps(this->recorder()).spmv(A, tx, ty, trans, BLASScalar(alpha),
                                                 BLASScalar(beta));
                this->recording->run().value();

                this->expect_interleaved(read_tensor<T>(wide_y), expected, K_SAMPLE_TOLERANCE);
            });
        }

        TYPED_TEST(Medium_SparseSpmvTest, APairStoredTwiceCountsTwice)
        {
            using T = typename TypeParam::Value;
            using Wide = std::vector<std::complex<double>>;

            // Row 0 stores column 1 twice, as 2 and as 3; row 1 stores 4 in column 2.
            const std::vector<int64_t> offsets = {0, 2, 3};
            const std::vector<int64_t> columns = {1, 1, 2};
            const Wide values = {{2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
            const std::complex<double> zero(0.0, 0.0);

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                this->set_mode(mode);
                const SparseMatrix A = this->matrix_of_arrays(2, 3, offsets, columns, values, base);
                const Tensor by_columns = this->vector_of({{1.0, 0.0}, {10.0, 0.0}, {100.0, 0.0}});
                const Tensor by_rows = this->vector_of({{1.0, 0.0}, {10.0, 0.0}});
                Tensor product = this->vector_of(Wide(2, zero));
                Tensor transposed = this->vector_of(Wide(3, zero));

                SparseOps ops(this->recorder());
                ops.spmv(A, by_columns, product);
                ops.spmv(A, by_rows, transposed, Core::Transpose::Trans);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(product), {{50.0, 0.0}, {400.0, 0.0}});
                this->expect_equal(read_tensor<T>(transposed), {zero, {5.0, 0.0}, {40.0, 0.0}});
            });
        }

        /**
         * @brief Runs the HALF products whose sum a HALF accumulator would round away.
         */
        class Medium_SparseSpmvHalf : public Medium_SparseSpmvTest<SparseCase<sycl::half, int64_t>>
        {

        };

        TEST_F(Medium_SparseSpmvHalf, ARowAndAColumnAreSummedInFloat32)
        {
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide ones(3, one);
            const Wide start(1, zero);

            // 2048 + 1 rounds back to 2048 in HALF, so a HALF sum of the three entries is 0.
            HostTriplets<sycl::half> row;
            row.rows = 1;
            row.cols = 3;
            row.row_indices = {0, 0, 0};
            row.column_indices = {0, 1, 2};
            row.values = {sycl::half(2048.0f), sycl::half(1.0f), sycl::half(-2048.0f)};

            HostTriplets<sycl::half> column;
            column.rows = 3;
            column.cols = 1;
            column.row_indices = {0, 1, 2};
            column.column_indices = {0, 0, 0};
            column.values = row.values;

            in_each_forced_mode([&](ExecutionMode mode) {
                expect_equal(run(mode, row, Core::IndexBase::Zero, Core::Transpose::NoTrans, one,
                                 ones, zero, start),
                             Wide(1, one));
                expect_equal(run(mode, column, Core::IndexBase::Zero, Core::Transpose::Trans, one,
                                 ones, zero, start),
                             Wide(1, one));
            });
        }

        /**
         * @brief Calls the transposed SYCL kernel with a block count the test chooses.
         */
        class Medium_SparseSpmvKernel : public SushiBLASTest,
                                        public ::testing::WithParamInterface<BlockRow>
        {
            protected:
                /** @brief Returns A^T * x from the SYCL kernel run with blocks row blocks. */
                [[nodiscard]] std::vector<float> transposed_product(const HostTriplets<float>& host,
                                                                    const std::vector<float>& x,
                                                                    std::size_t blocks)
                {
                    const Core::IndexBase base = Core::IndexBase::Zero;
                    const SparseBuilder builder(engine->tensor_factory());
                    const SparseMatrix A = builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<int32_t>(host.row_indices, base),
                        stored_indices<int32_t>(host.column_indices, base), host.values);
                    const Tensor tx = create_initialized_tensor<float>({host.rows}, x);
                    Tensor ty = engine->create_tensor({host.cols});
                    Tensor partials =
                        engine->create_tensor({static_cast<int64_t>(blocks) * host.cols});

                    const Kernels::CsrPattern<int32_t> pattern{
                        A.rows, A.cols, A.nnz(), 0, A.row_offsets.data_as<int32_t>(),
                        A.column_indices.data_as<int32_t>()};
                    Kernels::SYCL::spmv<float, int32_t>(
                        TestEnvironment::queue(), Core::Transpose::Trans, pattern,
                        A.values.data_as<float>(), 1.0f, tx.data_as<float>(), 1, 0.0f,
                        ty.data_as<float>(), 1, partials.data_as<float>(), blocks, {}).wait();
                    return read_tensor<float>(ty);
                }

                /** @brief Expects every element of a float result to equal its host expectation. */
                static void expect_floats_equal(const std::vector<float>& actual,
                                                const std::vector<std::complex<double>>& expected)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t j = 0; j < actual.size(); ++j)
                        EXPECT_EQ(static_cast<double>(actual[j]), expected[j].real())
                            << "column " << j;
                }
        };

        TEST_P(Medium_SparseSpmvKernel, TheRowBlocksCoverEveryEntryOnce)
        {
            // A row block that misses an entry or reads one twice changes the sum of column 0.
            const HostTriplets<float> host = block_partition_matrix();
            const Wide wide_x = cycling_vector(static_cast<std::size_t>(host.rows));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Wide expected = host_product(host, Core::Transpose::Trans, one, wide_x, zero,
                                               Wide(static_cast<std::size_t>(host.cols), zero));

            const std::vector<float> actual =
                transposed_product(host, floats_of(wide_x), GetParam().blocks);

            expect_floats_equal(actual, expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SparseSpmvKernel, ::testing::ValuesIn(block_rows()),
                                 block_row_name);
    } // namespace Test
} // namespace SushiBLAS
