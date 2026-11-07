/**************************************************************************/
/* test_sparse_spmm.cpp                                                   */
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
 * @file test_sparse_spmm.cpp
 * @brief Tests SparseOps::spmm column by column against host products, on both layouts.
 * @author Mustafa Garip
 */

#include <limits>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
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
        namespace
        {
            /** @brief A host vector of expectations or of operand values. */
            using Wide = std::vector<std::complex<double>>;

            /** @brief A dense host matrix as its columns, each one element per row. */
            using Columns = std::vector<Wide>;

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

            /// @brief Columns of the dense operands the integer-valued matrix multiplies.
            constexpr std::size_t K_EXACT_DENSE_COLUMNS = 5;

            /// @brief The two layouts a dense operand is stored in.
            constexpr Core::Layout K_BOTH_LAYOUTS[2] = {Core::Layout::RowMajor,
                                                        Core::Layout::ColumnMajor};

            /** @brief Asks a comparison to demand equal elements, where a double asks for a bound. */
            struct Exactly
            {

            };

            /// @brief The request for equal elements.
            constexpr Exactly K_EXACTLY{};

            /** @brief Returns count columns of rows elements, every one holding value. */
            [[nodiscard]] Columns filled(std::size_t rows, std::size_t count,
                                         std::complex<double> value)
            {
                return Columns(count, Wide(rows, value));
            }

            /** @brief Returns count columns of rows elements; element (j, r) is j + 1 + 10 * r. */
            [[nodiscard]] Columns sample_dense_columns(std::size_t rows, std::size_t count)
            {
                Columns columns(count);
                for (std::size_t r = 0; r < count; ++r)
                    columns[r] = counting(rows, 1.0 + 10.0 * static_cast<double>(r));
                return columns;
            }

            /**
             * @brief Returns count columns of rows elements; element (j, r) cycles through
             *        -2, -1, 0, 1, 2 along j, shifted by 2 * r.
             */
            [[nodiscard]] Columns cycling_columns(std::size_t rows, std::size_t count)
            {
                Columns columns(count, Wide(rows));
                for (std::size_t r = 0; r < count; ++r)
                {
                    for (std::size_t j = 0; j < rows; ++j)
                        columns[r][j] = std::complex<double>(
                            static_cast<double>((j + 2 * r) % 5) - 2.0, 0.0);
                }
                return columns;
            }
        } // namespace

        /**
         * @brief Builds a CSR matrix and dense operands of the engine's layout and runs spmm in a
         *        chosen execution mode.
         */
        template<typename TypeParam>
        class Medium_SparseSpmmTest : public SushiBLASTest
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief Skips the HALF suite on a device without the fp16 aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (std::is_same_v<T, sycl::half> &&
                        !TestEnvironment::queue().get_device().has(
                            sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }

                /** @brief Runs check once on an engine of each layout, under a trace that names it. */
                template<typename Check>
                void in_each_layout(Check&& check)
                {
                    for (const Core::Layout layout : K_BOTH_LAYOUTS)
                    {
                        SCOPED_TRACE(layout == Core::Layout::RowMajor ? "row-major"
                                                                       : "column-major");
                        reinit_engine(layout);
                        check();
                    }
                }

                /** @brief Returns the CSR matrix of host triplets, stored with base. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<T>& host, Core::IndexBase base)
                {
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<Index>(host.row_indices, base),
                        stored_indices<Index>(host.column_indices, base), host.values, base);
                }

                /**
                 * @brief Returns a rank-2 tensor of the engine's layout holding a host matrix.
                 * @param columns At least one column of at least one element.
                 */
                [[nodiscard]] Tensor matrix_of(const Columns& columns)
                {
                    const std::size_t rows = columns.front().size();
                    const std::size_t count = columns.size();
                    const bool row_major = engine->layout() == Core::Layout::RowMajor;

                    std::vector<T> host(rows * count);
                    for (std::size_t r = 0; r < count; ++r)
                    {
                        for (std::size_t i = 0; i < rows; ++i)
                            host[row_major ? i * count + r : i + r * rows] =
                                narrowed<T>(columns[r][i]);
                    }
                    return create_initialized_tensor<T>(
                        {static_cast<int64_t>(rows), static_cast<int64_t>(count)}, host);
                }

                /** @brief Returns the columns of a rank-2 tensor of the engine's layout. */
                [[nodiscard]] std::vector<std::vector<T>> columns_of(const Tensor& matrix)
                {
                    const std::size_t rows = static_cast<std::size_t>(matrix.shape[0]);
                    const std::size_t count = static_cast<std::size_t>(matrix.shape[1]);
                    const bool row_major = matrix.layout == Core::Layout::RowMajor;
                    const std::vector<T> host = read_tensor<T>(matrix);

                    std::vector<std::vector<T>> columns(count, std::vector<T>(rows));
                    for (std::size_t r = 0; r < count; ++r)
                    {
                        for (std::size_t i = 0; i < rows; ++i)
                            columns[r][i] = host[row_major ? i * count + r : i + r * rows];
                    }
                    return columns;
                }

                /** @brief Returns C after C = alpha * op(A) * B + beta * C recorded in mode. */
                [[nodiscard]] std::vector<std::vector<T>> run(ExecutionMode mode,
                                                              const HostTriplets<T>& host,
                                                              Core::IndexBase base,
                                                              Core::Transpose trans,
                                                              std::complex<double> alpha,
                                                              const Columns& b,
                                                              std::complex<double> beta,
                                                              const Columns& c)
                {
                    set_mode(mode);
                    const SparseMatrix A = build(host, base);
                    const Tensor B = matrix_of(b);
                    Tensor C = matrix_of(c);
                    SushiBLAS::SparseOps(recorder()).spmm(A, B, C, trans, BLASScalar(alpha),
                                                       BLASScalar(beta));
                    recording->run().value();
                    return columns_of(C);
                }

                /** @brief Expects every element within tolerance of its host expectation. */
                static void expect_within(const std::vector<T>& actual, const Wide& expected,
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
                static void expect_within(const std::vector<T>& actual, const Wide& expected,
                                          Exactly)
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
                 * @brief Expects the product on both layouts and in both modes to hold expected.
                 * @param c The value of C before the call.
                 * @param bound A tolerance in elements, or K_EXACTLY for equal elements.
                 */
                template<typename Bound>
                void expect_columns(const HostTriplets<T>& host, Core::IndexBase base,
                                    Core::Transpose trans, std::complex<double> alpha,
                                    const Columns& b, std::complex<double> beta, const Columns& c,
                                    const Columns& expected, Bound bound)
                {
                    in_each_layout([&] {
                        in_each_forced_mode([&](ExecutionMode mode) {
                            const std::vector<std::vector<T>> actual =
                                run(mode, host, base, trans, alpha, b, beta, c);
                            ASSERT_EQ(actual.size(), expected.size());
                            for (std::size_t r = 0; r < actual.size(); ++r)
                            {
                                SCOPED_TRACE(r);
                                expect_within(actual[r], expected[r], bound);
                            }
                        });
                    });
                }

                /**
                 * @brief Expects the product to equal host_product of each column of B and C.
                 * @param c The value of C before the call; pass zeros to the host when beta is 0.
                 * @param bound A tolerance in elements, or K_EXACTLY for equal elements.
                 */
                template<typename Bound>
                void expect_host_columns(const HostTriplets<T>& host, Core::IndexBase base,
                                         Core::Transpose trans, std::complex<double> alpha,
                                         const Columns& b, std::complex<double> beta,
                                         const Columns& c, Bound bound)
                {
                    Columns expected(b.size());
                    for (std::size_t r = 0; r < b.size(); ++r)
                        expected[r] = host_product(host, trans, alpha, b[r], beta, c[r]);

                    expect_columns(host, base, trans, alpha, b, beta, c, expected, bound);
                }
        };

        SB_TYPED_TEST_SUITE(Medium_SparseSpmmTest, SparseCases);

        TYPED_TEST(Medium_SparseSpmmTest, EveryColumnOfTheSampleProductMatchesTheHost)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> host = sample_matrix<T>();
            const std::complex<double> alpha(2.0, 0.0);
            const std::complex<double> beta(-1.0, 0.0);
            const std::complex<double> one(1.0, 0.0);

            in_each_transpose([&](Core::Transpose trans) {
                const bool transposed = Core::is_transposed(trans);
                const std::size_t inputs = transposed ? 4 : 5;
                const std::size_t outputs = transposed ? 5 : 4;

                // Element (j, r) of B is j + 1 + 10 * r.
                const Columns b = sample_dense_columns(inputs, 3);

                in_each_index_base([&](Core::IndexBase base) {
                    this->expect_host_columns(host, base, trans, alpha, b, beta,
                                              filled(outputs, 3, one), K_SAMPLE_TOLERANCE);
                });
            });
        }

        TYPED_TEST(Medium_SparseSpmmTest, ConjTransConjugatesAComplexMatrixAndNothingElse)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> host = sample_matrix<T>();
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const Core::IndexBase base = Core::IndexBase::Zero;

            // Column 0 of B selects row 0 of A: 1 + 2i in column 0 and 2 - i in column 3.
            Columns b = filled(4, 2, zero);
            b[0][0] = conj_selector<T>();
            const Columns c = filled(5, 2, zero);
            const Wide untouched(5, zero);

            this->expect_columns(host, base, Core::Transpose::Trans, one, b, zero, c,
                                 {conj_probe_product<T>(false), untouched}, K_EXACTLY);
            this->expect_columns(host, base, Core::Transpose::ConjTrans, one, b, zero, c,
                                 {conj_probe_product<T>(true), untouched}, K_EXACTLY);
        }

        TYPED_TEST(Medium_SparseSpmmTest, ABetaOfZeroDoesNotReadC)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> host = sample_matrix<T>();
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            const double nan = std::numeric_limits<double>::quiet_NaN();

            in_plain_and_transposed([&](Core::Transpose trans) {
                const bool transposed = Core::is_transposed(trans);
                const std::size_t inputs = transposed ? 4 : 5;
                const std::size_t outputs = transposed ? 5 : 4;

                const Columns b = {counting(inputs, 1.0), counting(inputs, 11.0)};
                Columns expected(b.size());
                for (std::size_t r = 0; r < b.size(); ++r)
                    expected[r] = host_product(host, trans, one, b[r], zero, Wide(outputs, zero));

                this->expect_columns(host, Core::IndexBase::Zero, trans, one, b, zero,
                                     filled(outputs, 2, std::complex<double>(nan, nan)), expected,
                                     K_SAMPLE_TOLERANCE);
            });
        }

        TYPED_TEST(Medium_SparseSpmmTest, IntegerDataIsExactInBothModesForEveryTranspose)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> host = integer_valued_matrix<T>(K_EXACT_ROWS, K_EXACT_COLS,
                                                                  K_EXACT_PER_ROW, K_EXACT_SEED);
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> minus_one(-1.0, 0.0);

            in_each_transpose([&](Core::Transpose trans) {
                const bool transposed = Core::is_transposed(trans);
                const std::size_t inputs = static_cast<std::size_t>(transposed ? host.rows
                                                                               : host.cols);
                const std::size_t outputs = static_cast<std::size_t>(transposed ? host.cols
                                                                                : host.rows);

                // Element (j, r) of B cycles through -2, -1, 0, 1, 2 along j, shifted by 2 * r.
                const Columns b = cycling_columns(inputs, K_EXACT_DENSE_COLUMNS);

                this->expect_host_columns(host, Core::IndexBase::Zero, trans, one, b, minus_one,
                                          filled(outputs, K_EXACT_DENSE_COLUMNS, one), K_EXACTLY);
            });
        }

        TYPED_TEST(Medium_SparseSpmmTest, DenseOperandsWithoutColumnsRecordOneTaskAndGiveAnEmptyC)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> host = sample_matrix<T>();

            this->in_each_layout([&] {
                in_each_forced_mode([&](ExecutionMode mode) {
                    this->set_mode(mode);
                    const SparseMatrix A = this->build(host, Core::IndexBase::Zero);
                    const Tensor B = this->engine->create_tensor({5, 0}, sushiblas_dtype<T>());
                    Tensor C = this->engine->create_tensor({4, 0}, sushiblas_dtype<T>());

                    SparseOps(this->recorder()).spmm(A, B, C);

                    EXPECT_EQ(this->recording->recorded_count(), 1u);
                    EXPECT_TRUE(this->recording->run_succeeds());
                    EXPECT_EQ(C.num_elements, int64_t{0});
                });
            });
        }

        /**
         * @brief Runs the HALF product whose column sum a HALF workspace would round away.
         */
        class Medium_SparseSpmmHalf : public Medium_SparseSpmmTest<SparseCase<sycl::half, int64_t>>
        {

        };

        TEST_F(Medium_SparseSpmmHalf, ATransposedColumnIsSummedInFloat32)
        {
            // 2048 + 1 rounds back to 2048 in HALF, so a HALF sum of the three entries is 0.
            HostTriplets<sycl::half> column;
            column.rows = 3;
            column.cols = 1;
            column.row_indices = {0, 1, 2};
            column.column_indices = {0, 0, 0};
            column.values = {sycl::half(2048.0f), sycl::half(1.0f), sycl::half(-2048.0f)};

            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);

            expect_columns(column, Core::IndexBase::Zero, Core::Transpose::Trans, one,
                           filled(3, 2, one), zero, filled(1, 2, zero), filled(1, 2, one),
                           K_EXACTLY);
        }
    } // namespace Test
} // namespace SushiBLAS
