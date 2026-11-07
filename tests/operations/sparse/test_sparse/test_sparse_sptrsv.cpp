/**************************************************************************/
/* test_sparse_sptrsv.cpp                                                 */
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
 * @file test_sparse_sptrsv.cpp
 * @brief Tests SparseOps::sptrsv against triangular systems whose solution is known.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <algorithm>
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
#include <SushiBLAS/sparse_assembly/sparse_assembly.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/sptrsv.hpp>
#include "support/forced_mode_checks.hpp"
#include "support/sparse_test_sweeps.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief A host vector of expectations or of operand values. */
            using Wide = std::vector<std::complex<double>>;

            /// @brief Absolute tolerance: 128 FLOAT32 roundings of a value of order one,
            /// for an element of a hand-worked solution.
            constexpr double K_SOLVE_TOLERANCE = absolute_tolerance<float>(128);

            /// @brief Absolute tolerance: 16 FLOAT32 roundings of a value of order one,
            /// below the 2.4e-4 a HALF-carried solution misses by.
            constexpr double K_HALF_CARRY_TOLERANCE = absolute_tolerance<float>(16);

            /// @brief Order of the bidiagonal system; above one work-group on every device.
            constexpr int64_t K_LONG_ORDER = 5000;

            /// @brief Seed of the solution of the bidiagonal system.
            constexpr std::uint32_t K_LONG_SEED = 20261007u;

            /** @brief Returns the complex doubles of a list of real values. */
            [[nodiscard]] Wide reals(const std::vector<double>& values)
            {
                Wide wide(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    wide[i] = std::complex<double>(values[i], 0.0);
                return wide;
            }

            /**
             * @brief Returns the 4 x 4 lower triangular sample L, listed out of row order.
             * @param imaginary Whether a complex T gets the imaginary parts 1, -1, 2, 1, -1, 2,
             *        -3, 1 on the entries 2, 1, 3, 4, 5, 6, 7, 8.
             * @return Rows (2), (1, 3), (., 4, 5), (6, ., 7, 8); L * (1, 2, 3, 4) is
             *         (2, 7, 23, 59) and L^T * (1, 2, 3, 4) is (28, 18, 43, 32).
             */
            template<typename T>
            [[nodiscard]] HostTriplets<T> lower_sample(bool imaginary)
            {
                HostTriplets<T> m;
                m.rows = 4;
                m.cols = 4;
                m.row_indices = {3, 1, 0, 2, 3, 1, 3, 2};
                m.column_indices = {3, 0, 0, 2, 0, 1, 2, 1};

                const double real_parts[8] = {8, 1, 2, 5, 6, 3, 7, 4};
                const double imaginary_parts[8] = {1, -1, 1, -1, 2, 2, -3, 1};
                for (std::size_t k = 0; k < 8; ++k)
                    m.values.push_back(narrowed<T>(
                        std::complex<double>(real_parts[k], imaginary ? imaginary_parts[k] : 0.0)));
                return m;
            }

            /** @brief Returns the transpose of a matrix: every entry with its indices exchanged. */
            template<typename T>
            [[nodiscard]] HostTriplets<T> transpose_of(const HostTriplets<T>& host)
            {
                HostTriplets<T> m = host;
                std::swap(m.rows, m.cols);
                std::swap(m.row_indices, m.column_indices);
                return m;
            }

            /** @brief Returns a matrix with one more entry. */
            template<typename T>
            [[nodiscard]] HostTriplets<T> with_entry(HostTriplets<T> host, int64_t row,
                                                     int64_t column, std::complex<double> value)
            {
                host.row_indices.push_back(row);
                host.column_indices.push_back(column);
                host.values.push_back(narrowed<T>(value));
                return host;
            }

            /** @brief Returns a matrix whose every stored diagonal entry is NaN. */
            template<typename T>
            [[nodiscard]] HostTriplets<T> with_nan_diagonal(HostTriplets<T> host)
            {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                for (std::size_t k = 0; k < host.values.size(); ++k)
                {
                    if (host.row_indices[k] == host.column_indices[k])
                        host.values[k] = narrowed<T>(std::complex<double>(nan, nan));
                }
                return host;
            }

            /** @brief Returns a matrix with every stored value conjugated. */
            template<typename T>
            [[nodiscard]] HostTriplets<T> conjugated(HostTriplets<T> host)
            {
                for (T& value : host.values)
                    value = narrowed<T>(std::conj(widened(value)));
                return host;
            }

            /** @brief Holds a lower bidiagonal system of integers and its solution. */
            template<typename T>
            struct LongSystem
            {
                HostTriplets<T> lower; /**< The matrix: 2 on the diagonal, 1 below it. */
                Wide x;                /**< The solution, integers from -3 to 3. */
                Wide b;                /**< The right-hand side, 2 * x[i] + x[i - 1]. */
            };

            /** @brief Returns the bidiagonal system of order K_LONG_ORDER a fixed seed draws. */
            template<typename T>
            [[nodiscard]] LongSystem<T> long_bidiagonal_system()
            {
                std::mt19937 generator(K_LONG_SEED);
                std::uniform_int_distribution<int> draw(-3, 3);

                LongSystem<T> system;
                system.lower.rows = K_LONG_ORDER;
                system.lower.cols = K_LONG_ORDER;
                system.x.resize(static_cast<std::size_t>(K_LONG_ORDER));
                system.b.resize(system.x.size());
                for (int64_t i = 0; i < K_LONG_ORDER; ++i)
                {
                    const std::size_t at = static_cast<std::size_t>(i);
                    system.x[at] = std::complex<double>(static_cast<double>(draw(generator)), 0.0);
                    system.b[at] =
                        2.0 * system.x[at] + (i > 0 ? system.x[at - 1] : std::complex<double>());

                    system.lower.row_indices.push_back(i);
                    system.lower.column_indices.push_back(i);
                    system.lower.values.push_back(narrowed<T>({2.0, 0.0}));
                    if (i > 0)
                    {
                        system.lower.row_indices.push_back(i);
                        system.lower.column_indices.push_back(i - 1);
                        system.lower.values.push_back(narrowed<T>({1.0, 0.0}));
                    }
                }
                return system;
            }
        } // namespace

        /**
         * @brief Builds triangular CSR matrices from host triplets and runs sptrsv in a chosen
         *        execution mode.
         */
        template<typename TypeParam>
        class Medium_SparseSptrsvTest : public SushiBLASTest
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

                /** @brief Returns b after op(A) * x = b is recorded in mode and solved in place. */
                [[nodiscard]] std::vector<T> solve(ExecutionMode mode, const SparseMatrix& A,
                                                   Core::Uplo uplo, Core::Transpose trans,
                                                   Core::Diag diag, const Wide& b)
                {
                    set_mode(mode);
                    Tensor tb = vector_of(b);
                    SushiBLAS::SparseOps(recorder()).sptrsv(A, tb, uplo, trans, diag);
                    recording->run().value();
                    return read_tensor<T>(tb);
                }

                /** @brief Returns the solution of op(A) * x = b for the A host triplets give. */
                [[nodiscard]] std::vector<T> run(ExecutionMode mode, const HostTriplets<T>& host,
                                                 Core::IndexBase base, Core::Uplo uplo,
                                                 Core::Transpose trans, Core::Diag diag,
                                                 const Wide& b)
                {
                    return solve(mode, build(host, base), uplo, trans, diag, b);
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

                /** @brief Returns the elements of a result widened to complex doubles. */
                [[nodiscard]] static Wide widened_vector(const std::vector<T>& values)
                {
                    Wide wide(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        wide[i] = widened(values[i]);
                    return wide;
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
        };

        SB_TYPED_TEST_SUITE(Medium_SparseSptrsvTest, SparseCases);

        SB_TYPED_TEST_SUITE(Medium_SparseSptrsvTest, SparseCases);

        /**
         * @brief The sptrsv fixture for the routines that read the imaginary part; a real element
         *        type skips every test of it.
         */
        template<typename TypeParam>
        class Medium_SparseSptrsvComplexTest : public Medium_SparseSptrsvTest<TypeParam>
        {
            protected:
                /** @brief Skips the test when the element type is real. */
                void SetUp() override
                {
                    Medium_SparseSptrsvTest<TypeParam>::SetUp();
                    if (!Core::is_complex_v<typename TypeParam::Value>)
                        GTEST_SKIP() << "the routine under test reads an imaginary part";
                }
        };

        SB_TYPED_TEST_SUITE(Medium_SparseSptrsvComplexTest, SparseCases);

        /**
         * @brief Assembles the triplets of a lower triangle and solves on the host with the
         *        reference kernel, so no runtime and no queue run.
         */
        template<typename TypeParam>
        class Small_SparseSptrsvTripletOrder : public ::testing::Test
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief Returns x after the reference kernel solves op(A) * x = b for the lower triangle A. */
                [[nodiscard]] static std::vector<T> solve(const HostTriplets<T>& host,
                                                          Core::Transpose trans, const Wide& b)
                {
                    const Core::IndexBase base = Core::IndexBase::Zero;
                    const Sparse::HostCsr<Index, T> csr = Sparse::assemble_csr<Index, T>(
                        host.rows, host.cols, Core::first_index(base),
                        stored_indices<Index>(host.row_indices, base),
                        stored_indices<Index>(host.column_indices, base), host.values);
                    const Kernels::CsrPattern<Index> pattern{
                        host.rows, host.cols, static_cast<int64_t>(csr.values.size()),
                        Core::first_index(base), csr.row_offsets.data(),
                        csr.column_indices.data()};

                    std::vector<T> x(b.size());
                    for (std::size_t i = 0; i < b.size(); ++i)
                        x[i] = narrowed<T>(b[i]);
                    Kernels::Reference::sptrsv(Core::Uplo::Lower, trans, Core::Diag::NonUnit,
                                               pattern, csr.values.data(), x.data());
                    return x;
                }

                /** @brief Expects two solutions to hold the same bits in every element. */
                static void expect_same_bits(const std::vector<T>& first,
                                             const std::vector<T>& second)
                {
                    ASSERT_EQ(first.size(), second.size());
                    EXPECT_EQ(std::memcmp(first.data(), second.data(), first.size() * sizeof(T)),
                              0);
                }
        };

        SB_TYPED_TEST_SUITE(Small_SparseSptrsvTripletOrder, SparseCases);

        TYPED_TEST(Medium_SparseSptrsvTest, BothTrianglesSolveBothSystemsOfTheSample)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> lower = lower_sample<T>(false);
            const HostTriplets<T> upper = transpose_of(lower);
            const Wide x = counting(4, 1.0);
            const Wide plain = reals({2, 7, 23, 59});
            const Wide transposed = reals({28, 18, 43, 32});
            const Core::Diag diag = Core::Diag::NonUnit;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                this->expect_near(this->run(mode, lower, base, Core::Uplo::Lower,
                                            Core::Transpose::NoTrans, diag, plain),
                                  x, K_SOLVE_TOLERANCE);
                this->expect_near(this->run(mode, lower, base, Core::Uplo::Lower,
                                            Core::Transpose::Trans, diag, transposed),
                                  x, K_SOLVE_TOLERANCE);
                this->expect_near(this->run(mode, upper, base, Core::Uplo::Upper,
                                            Core::Transpose::NoTrans, diag, transposed),
                                  x, K_SOLVE_TOLERANCE);
                this->expect_near(this->run(mode, upper, base, Core::Uplo::Upper,
                                            Core::Transpose::Trans, diag, plain),
                                  x, K_SOLVE_TOLERANCE);
            });
        }

        TYPED_TEST(Medium_SparseSptrsvTest, AUnitDiagonalIsNotRead)
        {
            using T = typename TypeParam::Value;

            // Every stored diagonal entry is NaN, so reading one would poison the solution.
            const HostTriplets<T> lower = with_nan_diagonal(lower_sample<T>(false));
            const Wide x = counting(4, 1.0);

            in_each_forced_mode([&](ExecutionMode mode) {
                this->expect_near(this->run(mode, lower, Core::IndexBase::Zero, Core::Uplo::Lower,
                                            Core::Transpose::NoTrans, Core::Diag::Unit,
                                            reals({1, 3, 11, 31})),
                                  x, K_SOLVE_TOLERANCE);
                this->expect_near(this->run(mode, lower, Core::IndexBase::Zero, Core::Uplo::Lower,
                                            Core::Transpose::Trans, Core::Diag::Unit,
                                            reals({27, 14, 31, 4})),
                                  x, K_SOLVE_TOLERANCE);
            });
        }

        TYPED_TEST(Medium_SparseSptrsvTest, TheOtherTriangleIsNotRead)
        {
            using T = typename TypeParam::Value;

            const std::complex<double> nan(std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::quiet_NaN());
            const HostTriplets<T> poisoned = with_entry(
                with_entry(with_entry(lower_sample<T>(false), 0, 1, nan), 0, 3, nan), 1, 2, nan);
            const Wide x = counting(4, 1.0);

            in_each_forced_mode([&](ExecutionMode mode) {
                this->expect_near(this->run(mode, poisoned, Core::IndexBase::Zero,
                                            Core::Uplo::Lower, Core::Transpose::NoTrans,
                                            Core::Diag::NonUnit, reals({2, 7, 23, 59})),
                                  x, K_SOLVE_TOLERANCE);
                this->expect_near(this->run(mode, poisoned, Core::IndexBase::Zero,
                                            Core::Uplo::Lower, Core::Transpose::Trans,
                                            Core::Diag::NonUnit, reals({28, 18, 43, 32})),
                                  x, K_SOLVE_TOLERANCE);
            });
        }

        TYPED_TEST(Small_SparseSptrsvTripletOrder, TheOrderOfTheTripletsDoesNotChangeTheBits)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> listed = lower_sample<T>(true);
            HostTriplets<T> reversed = listed;
            std::reverse(reversed.row_indices.begin(), reversed.row_indices.end());
            std::reverse(reversed.column_indices.begin(), reversed.column_indices.end());
            std::reverse(reversed.values.begin(), reversed.values.end());
            const Wide b = reals({1, 2, 3, 4});

            in_plain_and_transposed([&](Core::Transpose trans) {
                this->expect_same_bits(this->solve(listed, trans, b),
                                       this->solve(reversed, trans, b));
            });
        }

        TYPED_TEST(Medium_SparseSptrsvTest, ADiagonalStoredAsTwoEntriesIsTheirSum)
        {
            // Row 0 stores its diagonal as 1.5 and 0.5; row 1 stores 1 and the diagonal 3.
            const std::vector<int64_t> offsets = {0, 2, 4};
            const std::vector<int64_t> columns = {0, 0, 0, 1};
            const Wide values = reals({1.5, 0.5, 1, 3});
            const Wide x = counting(2, 1.0);

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                const SparseMatrix A = this->matrix_of_arrays(2, 2, offsets, columns, values, base);
                this->expect_near(this->solve(mode, A, Core::Uplo::Lower, Core::Transpose::NoTrans,
                                              Core::Diag::NonUnit, reals({2, 7})),
                                  x, K_SOLVE_TOLERANCE);
                this->expect_near(this->solve(mode, A, Core::Uplo::Lower, Core::Transpose::Trans,
                                              Core::Diag::NonUnit, reals({4, 6})),
                                  x, K_SOLVE_TOLERANCE);
            });
        }

        TYPED_TEST(Medium_SparseSptrsvComplexTest, ConjTransSolvesAgainstTheConjugateTranspose)
        {
            using T = typename TypeParam::Value;

            const HostTriplets<T> lower = lower_sample<T>(true);
            const HostTriplets<T> conjugate = conjugated(lower);
            const Wide b = {{1.0, 1.0}, {2.0, -1.0}, {3.0, 2.0}, {4.0, 0.0}};

            in_each_forced_mode([&](ExecutionMode mode) {
                const std::vector<T> conjugated_solution =
                    this->run(mode, lower, Core::IndexBase::Zero, Core::Uplo::Lower,
                              Core::Transpose::ConjTrans, Core::Diag::NonUnit, b);
                const std::vector<T> of_conjugate =
                    this->run(mode, conjugate, Core::IndexBase::Zero, Core::Uplo::Lower,
                              Core::Transpose::Trans, Core::Diag::NonUnit, b);
                const std::vector<T> plain =
                    this->run(mode, lower, Core::IndexBase::Zero, Core::Uplo::Lower,
                              Core::Transpose::Trans, Core::Diag::NonUnit, b);

                this->expect_near(conjugated_solution, this->widened_vector(of_conjugate),
                                  K_SOLVE_TOLERANCE);

                // Element 3 is 4 / (8 - i) against 4 / (8 + i): the imaginary parts are opposite.
                ASSERT_EQ(plain.size(), 4u);
                EXPECT_LT(widened(conjugated_solution[3]).imag() * widened(plain[3]).imag(), 0.0);
            });
        }

        TYPED_TEST(Medium_SparseSptrsvTest, SystemsOfOrderOneAndZeroSolve)
        {
            using T = typename TypeParam::Value;

            HostTriplets<T> one_entry;
            one_entry.rows = 1;
            one_entry.cols = 1;
            one_entry.row_indices = {0};
            one_entry.column_indices = {0};
            one_entry.values = {narrowed<T>({4.0, 0.0})};
            const HostTriplets<T> empty;

            in_each_forced_mode([&](ExecutionMode mode) {
                in_each_triangle([&](Core::Uplo uplo) {
                    this->expect_equal(this->run(mode, one_entry, Core::IndexBase::Zero, uplo,
                                                 Core::Transpose::NoTrans, Core::Diag::NonUnit,
                                                 reals({2})),
                                       reals({0.5}));
                });

                // A system of order zero records one task and runs; b has no element to check.
                this->set_mode(mode);
                const SparseMatrix A = this->build(empty, Core::IndexBase::Zero);
                Tensor b = this->engine->create_tensor({0}, sushiblas_dtype<T>());
                SparseOps(this->recorder()).sptrsv(A, b, Core::Uplo::Lower);
                EXPECT_EQ(this->recording->recorded_count(), 1u);
                EXPECT_TRUE(this->recording->run_succeeds());
            });
        }

        TYPED_TEST(Medium_SparseSptrsvTest, ALongBidiagonalSystemOfIntegersIsSolvedExactly)
        {
            using T = typename TypeParam::Value;

            // The diagonal is 2 and the subdiagonal 1, so x[i] = (b[i] - x[i - 1]) / 2 is exact.
            const LongSystem<T> system = long_bidiagonal_system<T>();

            in_each_forced_mode([&](ExecutionMode mode) {
                this->expect_equal(this->run(mode, system.lower, Core::IndexBase::Zero,
                                             Core::Uplo::Lower, Core::Transpose::NoTrans,
                                             Core::Diag::NonUnit, system.b),
                                   system.x);
            });
        }

        /**
         * @brief Runs the HALF system whose second step a HALF solution would get wrong.
         */
        class Medium_SparseSptrsvHalf : public Medium_SparseSptrsvTest<SparseCase<sycl::half, int64_t>>
        {

        };

        TEST_F(Medium_SparseSptrsvHalf, TheSolutionIsCarriedInFloat32)
        {
            // Rows (3) and (3, 1) with b = (1, 1): x[1] = 1 - 3 * x[0], and x[0] is a third.
            HostTriplets<sycl::half> lower;
            lower.rows = 2;
            lower.cols = 2;
            lower.row_indices = {0, 1, 1};
            lower.column_indices = {0, 0, 1};
            lower.values = {sycl::half(3.0f), sycl::half(3.0f), sycl::half(1.0f)};

            in_each_forced_mode([&](ExecutionMode mode) {
                const std::vector<sycl::half> x =
                    run(mode, lower, Core::IndexBase::Zero, Core::Uplo::Lower,
                        Core::Transpose::NoTrans, Core::Diag::NonUnit, reals({1, 1}));

                ASSERT_EQ(x.size(), 2u);
                EXPECT_NEAR(static_cast<double>(static_cast<float>(x[1])), 0.0,
                            K_HALF_CARRY_TOLERANCE);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
