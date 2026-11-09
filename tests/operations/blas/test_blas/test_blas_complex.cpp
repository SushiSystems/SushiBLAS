/**************************************************************************/
/* test_blas_complex.cpp                                                  */
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
 * @file test_blas_complex.cpp
 * @brief Tests the conjugating BLAS routines, CONJ_TRANS and their forwards on a real tensor,
 *        one row per layout, kernel and element type.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/tolerance.hpp"
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The type hand-written and integer test data is stated in.
            using Wide = std::complex<double>;

            /// @brief Seed of the data every random case draws.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Fills both parts of an element a call must neither read nor write.
            constexpr double K_POISON = 999.0;

            /// @brief Imaginary part given to a Hermitian diagonal, which a call must not read.
            constexpr double K_DIAGONAL_GARBAGE = 5.0;

            /// @brief Added to the diagonal of a triangular matrix, so a solve stays well scaled.
            constexpr double K_DIAGONAL_SHIFT = 4.0;

            /// @brief Real scalar of the Hermitian rank updates.
            constexpr double K_REAL_ALPHA = 1.5;

            /// @brief Real scalar of the prior result of a Hermitian rank-k update.
            constexpr double K_REAL_BETA = -0.75;

            /// @brief Rows of op(A) in the random GEMM, GEMV and TRSM cases.
            constexpr int64_t K_CASE_ROWS = 5;

            /// @brief Columns of op(A) and rows of op(B) in the random GEMM case.
            constexpr int64_t K_CASE_INNER = 4;

            /// @brief Columns of the result in the random GEMM, GEMV and TRSM cases.
            constexpr int64_t K_CASE_COLS = 6;

            /// @brief Order of the square matrices and length of the vectors in the random cases.
            constexpr int64_t K_CASE_ORDER = 7;

            /// @brief Matrices in the batched GEMM case.
            constexpr int64_t K_CASE_BATCH = 3;

            /// @brief Rounded operations on the path to one result of a hand case: a handful of
            ///        products and sums of small integers.
            constexpr std::size_t K_HAND_OPERATIONS = 8;

            /// @brief Rounded operations on the path to one result of two calls that sum
            ///        K_CASE_ORDER terms, each scaled once.
            constexpr std::size_t K_AGREEMENT_OPERATIONS = 2 * static_cast<std::size_t>(K_CASE_ORDER);

            /// @brief Ratio of error to rounding bound above which two calls disagree.
            constexpr double K_AGREEMENT_THRESHOLD = 16.0;

            /// @brief Both triangles a Hermitian or triangular call can name.
            const std::vector<Core::Uplo> K_TRIANGLES = {Core::Uplo::Upper, Core::Uplo::Lower};

            /// @brief Both kernels an operation has.
            const std::vector<ExecutionMode> K_BOTH_MODES = {ExecutionMode::ForceReference,
                                                             ExecutionMode::ForceSYCL};

            /// @brief The kernel the hand cases run: the reference computes them exactly.
            const std::vector<ExecutionMode> K_REFERENCE_MODE = {ExecutionMode::ForceReference};

            /// @brief The kernel the cases that compare the device with the reference run.
            const std::vector<ExecutionMode> K_DEVICE_MODE = {ExecutionMode::ForceSYCL};

            /// @brief The complex element types, the only ones a conjugation changes.
            const std::vector<Core::DataType> K_COMPLEX_TYPES = {Core::DataType::Complex32,
                                                                 Core::DataType::Complex64};

            /// @brief The element type of the forwards of a conjugating routine on a real tensor.
            const std::vector<Core::DataType> K_REAL_TYPE = {Core::DataType::Float32};

            /** @brief Holds the forms of op(A) and op(B) of one gemm call. */
            struct TransposePair
            {
                Core::Transpose a; /**< The form of op(A). */
                Core::Transpose b; /**< The form of op(B). */
            };

            /** @brief Holds the side and the triangle of one trsm call. */
            struct SideUplo
            {
                Core::Side side; /**< The side the triangular matrix stands on. */
                Core::Uplo uplo; /**< The triangle. */
            };

            /** @brief Holds the triangle and the form of A of one rank-k update. */
            struct HerkForm
            {
                Core::Uplo uplo;       /**< The triangle of C. */
                Core::Transpose trans; /**< The form of A. */
            };

            /// @brief The pairs of forms the conjugated gemm is compared on.
            const std::vector<TransposePair> K_GEMM_PAIRS = {
                {Core::Transpose::ConjTrans, Core::Transpose::NoTrans},
                {Core::Transpose::NoTrans, Core::Transpose::ConjTrans},
                {Core::Transpose::ConjTrans, Core::Transpose::ConjTrans},
                {Core::Transpose::ConjTrans, Core::Transpose::Trans},
                {Core::Transpose::Trans, Core::Transpose::ConjTrans}};

            /** @brief Returns every side and triangle. */
            [[nodiscard]] std::vector<SideUplo> all_sides_and_triangles()
            {
                std::vector<SideUplo> rows;
                for (const Core::Side side : {Core::Side::Left, Core::Side::Right})
                    for (const Core::Uplo uplo : K_TRIANGLES)
                        rows.push_back({side, uplo});
                return rows;
            }

            /** @brief Returns every triangle with each of the two forms. */
            [[nodiscard]] std::vector<HerkForm> herk_forms(Core::Transpose first,
                                                           Core::Transpose second)
            {
                std::vector<HerkForm> rows;
                for (const Core::Uplo uplo : K_TRIANGLES)
                    for (const Core::Transpose trans : {first, second})
                        rows.push_back({uplo, trans});
                return rows;
            }

            /** @brief Returns the tail of a row named by the forms of a gemm call. */
            [[nodiscard]] std::string axis_name(const TransposePair& pair)
            {
                return "_" + word(pair.a) + "_" + word(pair.b);
            }

            /** @brief Returns the tail of a row named by the side and triangle of a trsm call. */
            [[nodiscard]] std::string axis_name(const SideUplo& form)
            {
                return "_" + word(form.side) + "_" + word(form.uplo);
            }

            /** @brief Returns the tail of a row named by the triangle and form of a rank-k update. */
            [[nodiscard]] std::string axis_name(const HerkForm& form)
            {
                return "_" + word(form.uplo) + "_" + word(form.trans);
            }

            /** @brief Returns whether the test device computes in the component type of a data type. */
            [[nodiscard]] bool device_runs(Core::DataType type)
            {
                const sycl::device device = TestEnvironment::queue().get_device();
                if (type == Core::DataType::Complex64)
                    return device.has(sycl::aspect::fp64);
                return true;
            }

            /** @brief Returns value as a complex double; a real T has a zero imaginary part. */
            template<typename T>
            [[nodiscard]] Wide widened(const T& value)
            {
                if constexpr (Core::is_complex_v<T>)
                    return Detail::to_complex_double(value);
                else
                    return Wide(Detail::to_double(value), 0.0);
            }

            /** @brief Holds how far a result may lie from its expectation, in rounding terms. */
            struct Rounding
            {
                std::size_t operations; /**< Rounded operations on the path to one result. */
                double threshold;       /**< Ratio of error to bound above which a result fails. */
            };

            /// @brief The rounding of a hand case.
            constexpr Rounding K_HAND_ROUNDING = {K_HAND_OPERATIONS, TestSupport::K_DEFAULT_THRESHOLD};

            /// @brief The rounding of two calls over the same draws.
            constexpr Rounding K_AGREEMENT_ROUNDING = {K_AGREEMENT_OPERATIONS, K_AGREEMENT_THRESHOLD};

            /**
             * @brief Expects both parts of every element within the rounding bound of the
             *        expectation: operations * unit roundoff of the component type * magnitude,
             *        the magnitude being the largest of 1 and the expected modulus.
             */
            template<typename T>
            void expect_near(const std::vector<T>& actual, const std::vector<T>& expected,
                             const Rounding& rounding)
            {
                using R = Kernels::RealType<T>;

                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const Wide a = widened(actual[i]);
                    const Wide e = widened(expected[i]);
                    const double bound = static_cast<double>(rounding.operations) *
                                         TestSupport::unit_roundoff<R>() *
                                         std::max(1.0, std::abs(e));

                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        a.real(), e.real(), bound, rounding.threshold)))
                        << "real part of element " << i;
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        a.imag(), e.imag(), bound, rounding.threshold)))
                        << "imaginary part of element " << i;
                }
            }

            /** @brief Expects two host vectors to hold the same bytes, element by element. */
            template<typename T>
            void expect_same_bits(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0)
                    {
                        ADD_FAILURE() << "element " << i << " differs in its bits";
                        return;
                    }
                }
            }

            /** @brief Expects the diagonal of an n x n matrix to be exactly real. */
            template<typename T>
            void expect_real_diagonal(const std::vector<T>& matrix, int64_t n)
            {
                for (int64_t i = 0; i < n; ++i)
                    EXPECT_EQ(widened(matrix[static_cast<std::size_t>(i * n + i)]).imag(), 0.0)
                        << "diagonal element " << i;
            }

            /** @brief Returns real + imaginary * i in T; a real T drops the imaginary part. */
            template<typename T>
            [[nodiscard]] T make(double real, [[maybe_unused]] double imaginary) noexcept
            {
                using R = Kernels::RealType<T>;
                if constexpr (Core::is_complex_v<T>)
                    return T(static_cast<R>(real), static_cast<R>(imaginary));
                else
                    return static_cast<R>(real);
            }

            /** @brief Returns value as the scalar argument of a call. */
            template<typename T>
            [[nodiscard]] BLASScalar scalar(const T& value) noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return BLASScalar(std::complex<double>(static_cast<double>(value.real()),
                                                           static_cast<double>(value.imag())));
                else
                    return BLASScalar(static_cast<double>(value));
            }

            /** @brief Returns the scalar that multiplies the product or the update. */
            template<typename T>
            [[nodiscard]] T alpha() noexcept
            {
                return make<T>(1.5, -0.5);
            }

            /** @brief Returns the scalar that multiplies the prior result. */
            template<typename T>
            [[nodiscard]] T beta() noexcept
            {
                return make<T>(0.25, 0.75);
            }

            /** @brief Returns full with the triangle uplo does not name poisoned. */
            template<typename T>
            [[nodiscard]] std::vector<T> keep_triangle(std::vector<T> full, int64_t n,
                                                       Core::Uplo uplo)
            {
                const bool upper = (uplo == Core::Uplo::Upper);
                for (int64_t r = 0; r < n; ++r)
                {
                    for (int64_t c = 0; c < n; ++c)
                    {
                        const bool kept = upper ? (r <= c) : (r >= c);
                        if (!kept)
                            full[static_cast<std::size_t>(r * n + c)] = make<T>(K_POISON, K_POISON);
                    }
                }
                return full;
            }

            /** @brief Returns matrix with K_DIAGONAL_GARBAGE on its imaginary diagonal. */
            template<typename T>
            [[nodiscard]] std::vector<T> with_imaginary_diagonal(std::vector<T> matrix, int64_t n)
            {
                for (int64_t i = 0; i < n; ++i)
                    matrix[static_cast<std::size_t>(i * n + i)] += make<T>(0.0, K_DIAGONAL_GARBAGE);
                return matrix;
            }
        } // namespace

        /**
         * @brief Builds the operands of a BLAS call from seeded data, records the call and reads
         *        its result in row order, whatever the layout of the engine.
         */
        class BlasComplexCase : public SushiBLASTest
        {
            protected:
                /** @brief Holds the transpose an operand is passed with and how it is drawn. */
                struct OperandForm
                {
                    Core::Transpose trans;  /**< The flag the call receives. */
                    bool conjugate_values;  /**< Whether the drawn values are conjugated first. */
                };

                /** @brief Returns the form that passes trans over the values as drawn. */
                [[nodiscard]] static constexpr OperandForm as_given(Core::Transpose trans) noexcept
                {
                    return OperandForm{trans, false};
                }

                /** @brief Returns TRANS over conjugated values for CONJ_TRANS, else as_given. */
                [[nodiscard]] static constexpr OperandForm
                on_conjugated_copy(Core::Transpose trans) noexcept
                {
                    return Core::is_conjugated(trans) ? OperandForm{Core::Transpose::Trans, true}
                                                      : as_given(trans);
                }

                /** @brief Rebuilds the engine in layout, selects mode and restarts the data. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    layout_ = layout;
                    mode_ = mode;
                    reinit_engine(layout);
                    set_mode(mode);
                    restart_data();
                }

                /** @brief Makes the next draw repeat the first one. */
                void restart_data()
                {
                    generator_.seed(K_SEED);
                }

                /** @brief Returns count values with every part drawn from [-1, 1]. */
                template<typename T>
                [[nodiscard]] std::vector<T> random_values(int64_t count)
                {
                    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                    std::vector<T> values(static_cast<std::size_t>(count));
                    for (T& value : values)
                    {
                        const double real = distribution(generator_);
                        const double imaginary = distribution(generator_);
                        value = make<T>(real, imaginary);
                    }
                    return values;
                }

                /** @brief Returns random_values, conjugated when form asks for it. */
                template<typename T>
                [[nodiscard]] std::vector<T> operand_values(int64_t count, const OperandForm& form)
                {
                    std::vector<T> values = random_values<T>(count);
                    if (form.conjugate_values)
                        for (T& value : values)
                            value = Kernels::conjugate(value);
                    return values;
                }

                /** @brief Returns an n x n operand in row order whose diagonal is shifted. */
                template<typename T>
                [[nodiscard]] std::vector<T> triangular_values(int64_t n, const OperandForm& form)
                {
                    std::vector<T> values = operand_values<T>(n * n, form);
                    for (int64_t i = 0; i < n; ++i)
                        values[static_cast<std::size_t>(i * n + i)] +=
                            make<T>(K_DIAGONAL_SHIFT, 0.0);
                    return values;
                }

                /** @brief Returns a rank-1 tensor holding values. */
                template<typename T>
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    return tensor_of({static_cast<int64_t>(values.size())}, values);
                }

                /** @brief Returns a rows x cols tensor holding the row-ordered values. */
                template<typename T>
                [[nodiscard]] Tensor matrix_of(int64_t rows, int64_t cols,
                                               const std::vector<T>& row_ordered)
                {
                    return matrices_of({}, rows, cols, row_ordered);
                }

                /**
                 * @brief Returns a tensor of batch_axes x rows x cols holding the row-ordered
                 *        matrices, stored in the layout of the engine.
                 */
                template<typename T>
                [[nodiscard]] Tensor matrices_of(std::vector<int64_t> batch_axes, int64_t rows,
                                                 int64_t cols, const std::vector<T>& row_ordered)
                {
                    std::vector<T> stored(row_ordered.size());
                    const std::size_t per_matrix = static_cast<std::size_t>(rows * cols);
                    for (std::size_t base = 0; base < stored.size(); base += per_matrix)
                        for (int64_t r = 0; r < rows; ++r)
                            for (int64_t c = 0; c < cols; ++c)
                                stored[base + storage_offset(r, c, rows, cols)] =
                                    row_ordered[base + static_cast<std::size_t>(r * cols + c)];

                    batch_axes.push_back(rows);
                    batch_axes.push_back(cols);
                    return tensor_of(batch_axes, stored);
                }

                /** @brief Returns the elements of a tensor, each matrix in row order. */
                template<typename T>
                [[nodiscard]] std::vector<T> read_row_ordered(const Tensor& tensor) const
                {
                    const std::vector<T> stored = read_tensor<T>(tensor);
                    if (tensor.rank < 2)
                        return stored;

                    const int64_t rows = tensor.shape[tensor.rank - 2];
                    const int64_t cols = tensor.shape[tensor.rank - 1];
                    const std::size_t per_matrix = static_cast<std::size_t>(rows * cols);
                    std::vector<T> row_ordered(stored.size());
                    for (std::size_t base = 0; base < stored.size(); base += per_matrix)
                        for (int64_t r = 0; r < rows; ++r)
                            for (int64_t c = 0; c < cols; ++c)
                                row_ordered[base + static_cast<std::size_t>(r * cols + c)] =
                                    stored[base + storage_offset(r, c, rows, cols)];
                    return row_ordered;
                }

                /** @brief Returns the result of a dot product call over two drawn vectors. */
                template<typename T, typename Call>
                [[nodiscard]] std::vector<T> run_dot(Call call)
                {
                    Tensor x = vector_of(random_values<T>(K_CASE_ORDER));
                    Tensor y = vector_of(random_values<T>(K_CASE_ORDER));
                    Tensor result = vector_of(std::vector<T>(1));

                    SushiBLAS::BLASOps ops(recorder());
                    (ops.*call)(x, y, result);
                    recording->run().value();
                    return read_row_ordered<T>(result);
                }

                /** @brief Returns A after a rank-1 update call scaled by scale. */
                template<typename T, typename Call, typename Scalar>
                [[nodiscard]] std::vector<T> run_ger(Call call, Scalar scale)
                {
                    Tensor x = vector_of(random_values<T>(K_CASE_ROWS));
                    Tensor y = vector_of(random_values<T>(K_CASE_COLS));
                    Tensor A = matrix_of(K_CASE_ROWS, K_CASE_COLS,
                                         random_values<T>(K_CASE_ROWS * K_CASE_COLS));

                    SushiBLAS::BLASOps ops(recorder());
                    (ops.*call)(x, y, A, scale);
                    recording->run().value();
                    return read_row_ordered<T>(A);
                }

                /** @brief Returns y after a symmetric or Hermitian matrix-vector call. */
                template<typename T, typename Call>
                [[nodiscard]] std::vector<T> run_hemv(Call call, Core::Uplo uplo)
                {
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values<T>(K_CASE_ORDER * K_CASE_ORDER));
                    Tensor x = vector_of(random_values<T>(K_CASE_ORDER));
                    Tensor y = vector_of(random_values<T>(K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(recorder());
                    (ops.*call)(A, x, y, uplo, scalar(alpha<T>()), scalar(beta<T>()));
                    recording->run().value();
                    return read_row_ordered<T>(y);
                }

                /** @brief Returns A after a symmetric or Hermitian rank-1 update call. */
                template<typename T, typename Call>
                [[nodiscard]] std::vector<T> run_her(Call call, Core::Uplo uplo)
                {
                    Tensor x = vector_of(random_values<T>(K_CASE_ORDER));
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values<T>(K_CASE_ORDER * K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(recorder());
                    (ops.*call)(x, A, uplo, K_REAL_ALPHA);
                    recording->run().value();
                    return read_row_ordered<T>(A);
                }

                /** @brief Returns A after a symmetric or Hermitian rank-2 update call. */
                template<typename T, typename Call>
                [[nodiscard]] std::vector<T> run_her2(Call call, Core::Uplo uplo)
                {
                    Tensor x = vector_of(random_values<T>(K_CASE_ORDER));
                    Tensor y = vector_of(random_values<T>(K_CASE_ORDER));
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values<T>(K_CASE_ORDER * K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(recorder());
                    (ops.*call)(x, y, A, uplo, scalar(alpha<T>()));
                    recording->run().value();
                    return read_row_ordered<T>(A);
                }

                /** @brief Returns C after a symmetric or Hermitian rank-k update call. */
                template<typename T, typename Call>
                [[nodiscard]] std::vector<T> run_herk(Call call, Core::Uplo uplo,
                                                      Core::Transpose trans)
                {
                    const bool transposed = Core::is_transposed(trans);
                    const int64_t a_rows = transposed ? K_CASE_INNER : K_CASE_ORDER;
                    const int64_t a_cols = transposed ? K_CASE_ORDER : K_CASE_INNER;
                    Tensor A = matrix_of(a_rows, a_cols, random_values<T>(a_rows * a_cols));
                    Tensor C = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values<T>(K_CASE_ORDER * K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(recorder());
                    (ops.*call)(A, C, uplo, trans, K_REAL_ALPHA, K_REAL_BETA);
                    recording->run().value();
                    return read_row_ordered<T>(C);
                }

                /** @brief Returns C = alpha * op(A) * op(B) + beta * C over drawn operands. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_gemm(const OperandForm& a, const OperandForm& b,
                                                      const std::vector<int64_t>& batch_axes)
                {
                    int64_t batch = 1;
                    for (const int64_t axis : batch_axes)
                        batch *= axis;

                    const bool a_transposed = Core::is_transposed(a.trans);
                    const bool b_transposed = Core::is_transposed(b.trans);
                    const int64_t a_rows = a_transposed ? K_CASE_INNER : K_CASE_ROWS;
                    const int64_t a_cols = a_transposed ? K_CASE_ROWS : K_CASE_INNER;
                    const int64_t b_rows = b_transposed ? K_CASE_COLS : K_CASE_INNER;
                    const int64_t b_cols = b_transposed ? K_CASE_INNER : K_CASE_COLS;

                    Tensor A = matrices_of(batch_axes, a_rows, a_cols,
                                           operand_values<T>(batch * a_rows * a_cols, a));
                    Tensor B = matrices_of(batch_axes, b_rows, b_cols,
                                           operand_values<T>(batch * b_rows * b_cols, b));
                    Tensor C = matrices_of(batch_axes, K_CASE_ROWS, K_CASE_COLS,
                                           random_values<T>(batch * K_CASE_ROWS * K_CASE_COLS));

                    SushiBLAS::BLASOps(recorder()).gemm(A, B, C, a.trans, b.trans,
                                                     scalar(alpha<T>()), scalar(beta<T>()));
                    recording->run().value();
                    return read_row_ordered<T>(C);
                }

                /** @brief Returns y = alpha * op(A) * x + beta * y over drawn operands. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_gemv(const OperandForm& a)
                {
                    const bool transposed = Core::is_transposed(a.trans);
                    Tensor A = matrix_of(K_CASE_ROWS, K_CASE_COLS,
                                         operand_values<T>(K_CASE_ROWS * K_CASE_COLS, a));
                    Tensor x = vector_of(random_values<T>(transposed ? K_CASE_ROWS : K_CASE_COLS));
                    Tensor y = vector_of(random_values<T>(transposed ? K_CASE_COLS : K_CASE_ROWS));

                    SushiBLAS::BLASOps(recorder()).gemv(A, x, y, a.trans, scalar(alpha<T>()),
                                                     scalar(beta<T>()));
                    recording->run().value();
                    return read_row_ordered<T>(y);
                }

                /** @brief Returns x = op(A) * x for a drawn triangular A. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_trmv(Core::Uplo uplo, const OperandForm& a)
                {
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         triangular_values<T>(K_CASE_ORDER, a));
                    Tensor x = vector_of(random_values<T>(K_CASE_ORDER));

                    SushiBLAS::BLASOps(recorder()).trmv(A, x, uplo, a.trans, Core::Diag::NonUnit);
                    recording->run().value();
                    return read_row_ordered<T>(x);
                }

                /** @brief Returns the solution of op(A) * x = b for a drawn triangular A. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_trsv(Core::Uplo uplo, const OperandForm& a)
                {
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         triangular_values<T>(K_CASE_ORDER, a));
                    Tensor b = vector_of(random_values<T>(K_CASE_ORDER));

                    SushiBLAS::BLASOps(recorder()).trsv(A, b, uplo, a.trans, Core::Diag::NonUnit);
                    recording->run().value();
                    return read_row_ordered<T>(b);
                }

                /** @brief Returns the solution of a triangular system with a drawn A on side. */
                template<typename T>
                [[nodiscard]] std::vector<T> run_trsm(Core::Side side, Core::Uplo uplo,
                                                      const OperandForm& a)
                {
                    const int64_t order = (side == Core::Side::Left) ? K_CASE_ROWS : K_CASE_COLS;
                    Tensor A = matrix_of(order, order, triangular_values<T>(order, a));
                    Tensor B = matrix_of(K_CASE_ROWS, K_CASE_COLS,
                                         random_values<T>(K_CASE_ROWS * K_CASE_COLS));

                    SushiBLAS::BLASOps(recorder()).trsm(A, B, side, uplo, a.trans,
                                                     Core::Diag::NonUnit, scalar(alpha<T>()));
                    recording->run().value();
                    return read_row_ordered<T>(B);
                }

                /**
                 * @brief Expects call to give the same result on the reference and on the kernel
                 *        of the row, in the layout of the row.
                 */
                template<typename T, typename Call>
                void expect_modes_agree(Call&& call)
                {
                    const Core::Layout layout = layout_;
                    const ExecutionMode device_mode = mode_;

                    start(layout, ExecutionMode::ForceReference);
                    const std::vector<T> reference = call();
                    start(layout, device_mode);
                    const std::vector<T> device = call();

                    expect_near(device, reference, K_AGREEMENT_ROUNDING);
                }

                /** @brief Expects two calls over the same draws to agree in the engine of the row. */
                template<typename T, typename First, typename Second>
                void expect_calls_agree(First&& first, Second&& second)
                {
                    const std::vector<T> from_first = first();
                    restart_data();
                    const std::vector<T> from_second = second();

                    expect_near(from_first, from_second, K_AGREEMENT_ROUNDING);
                }

                /** @brief Expects two calls over the same draws to give the same bits. */
                template<typename T, typename First, typename Second>
                void expect_calls_give_the_same_bits(First&& first, Second&& second)
                {
                    const std::vector<T> from_first = first();
                    restart_data();
                    const std::vector<T> from_second = second();

                    expect_same_bits(from_first, from_second);
                }

            private:
                /** @brief Returns where element (r, c) of a rows x cols matrix is stored. */
                [[nodiscard]] std::size_t storage_offset(int64_t r, int64_t c, int64_t rows,
                                                         int64_t cols) const
                {
                    return static_cast<std::size_t>(
                        (engine->layout() == Core::Layout::RowMajor) ? r * cols + c
                                                                          : r + c * rows);
                }

                /** @brief Returns a tensor of dims holding stored in storage order. */
                template<typename T>
                [[nodiscard]] Tensor tensor_of(const std::vector<int64_t>& dims,
                                               const std::vector<T>& stored)
                {
                    Tensor tensor = engine->create_tensor(
                        SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                        sushiblas_dtype<T>());
                    TestEnvironment::queue()
                        .memcpy(tensor.data(), stored.data(), stored.size() * sizeof(T)).wait();
                    return tensor;
                }

                /// @brief The generator every draw advances.
                std::mt19937 generator_{K_SEED};

                /// @brief The layout the engine was last started in.
                Core::Layout layout_ = Core::Layout::RowMajor;

                /// @brief The kernel the engine was last started with.
                ExecutionMode mode_ = ExecutionMode::ForceReference;
        };

        /**
         * @brief Runs a case once per row: rebuilds the engine in the layout of the row, selects
         *        its kernel and skips a device that lacks the element type.
         * @tparam Axis NoAxis, or the type of the value that completes each row.
         */
        template<typename Axis>
        class ComplexRowCase : public BlasComplexCase, public ::testing::WithParamInterface<Row<Axis>>
        {
            protected:
                /** @brief Starts the engine of the row, or skips the case. */
                void SetUp() override
                {
                    BlasComplexCase::SetUp();
                    if (!device_runs(this->GetParam().type))
                        GTEST_SKIP() << "the device does not compute in this element type";
                    start(this->GetParam().layout, this->GetParam().mode);
                }
        };

        /** @brief Runs the hand cases and the comparisons that take one triangle. */
        class Medium_BlasComplexTriangle : public ComplexRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the rank-k update hand cases, one per triangle and form. */
        class Medium_BlasComplexHerk : public ComplexRowCase<HerkForm>
        {

        };

        /** @brief Runs the gemm comparison of a conjugated form against a conjugated copy. */
        class Medium_BlasComplexPair : public ComplexRowCase<TransposePair>
        {

        };

        /** @brief Runs the conjugated-copy comparisons that take no further axis. */
        class Medium_BlasComplexCopy : public ComplexRowCase<NoAxis>
        {

        };

        /** @brief Runs the conjugated-copy comparisons that take one triangle. */
        class Medium_BlasComplexCopyTriangle : public ComplexRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the conjugated-copy comparison of trsm on each side and triangle. */
        class Medium_BlasComplexCopySide : public ComplexRowCase<SideUplo>
        {

        };

        /** @brief Runs the comparison of the device with the reference, no further axis. */
        class Medium_BlasComplexAgree : public ComplexRowCase<NoAxis>
        {

        };

        /** @brief Runs the comparison of the device with the reference, one triangle. */
        class Medium_BlasComplexAgreeTriangle : public ComplexRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the comparison of the device with the reference for herk. */
        class Medium_BlasComplexAgreeHerk : public ComplexRowCase<HerkForm>
        {

        };

        /** @brief Runs the comparison of the device with the reference for trsm. */
        class Medium_BlasComplexAgreeSide : public ComplexRowCase<SideUplo>
        {

        };

        /** @brief Runs the forwards of the conjugating routines on a real tensor, no axis. */
        class Medium_BlasRealForwards : public ComplexRowCase<NoAxis>
        {

        };

        /** @brief Runs the forwards on a real tensor that take one triangle. */
        class Medium_BlasRealForwardsTriangle : public ComplexRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the forward of herk to syrk on a real tensor. */
        class Medium_BlasRealForwardsHerk : public ComplexRowCase<HerkForm>
        {

        };

        TEST_P(Medium_BlasComplexTriangle, HemvReadsOneTriangleAndNoImaginaryDiagonal)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const auto z = &make<T>;
                const std::vector<T> hermitian = {z(2, 0), z(1, 1), z(0, -1),
                                                  z(1, -1), z(3, 0), z(2, 1),
                                                  z(0, 1), z(2, -1), z(-1, 0)};
                const std::vector<T> x = {z(1, 0), z(0, 1), z(2, 0)};
                const std::vector<T> expected = {z(1, -1), z(5, 4), z(-1, 3)};

                Tensor A = matrix_of(3, 3, with_imaginary_diagonal(keep_triangle(hermitian, 3, uplo), 3));
                Tensor x_tensor = vector_of(x);
                Tensor y_tensor = vector_of(std::vector<T>(3));

                SushiBLAS::BLASOps(recorder()).hemv(A, x_tensor, y_tensor, uplo, 1.0, 0.0);
                recording->run().value();

                expect_near(read_row_ordered<T>(y_tensor), expected, K_HAND_ROUNDING);
            });
        }

        TEST_P(Medium_BlasComplexTriangle, HemvScalesAndAccumulates)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const auto z = &make<T>;
                const std::vector<T> hermitian = {z(2, 0), z(1, 1), z(0, -1),
                                                  z(1, -1), z(3, 0), z(2, 1),
                                                  z(0, 1), z(2, -1), z(-1, 0)};
                const std::vector<T> x = {z(1, 0), z(0, 1), z(2, 0)};
                const std::vector<T> y = {z(1, 0), z(0, 1), z(-1, 0)};
                const std::vector<T> expected = {z(4, 0), z(1, 11), z(-6, 2)};

                Tensor A = matrix_of(3, 3, with_imaginary_diagonal(keep_triangle(hermitian, 3, uplo), 3));
                Tensor x_tensor = vector_of(x);
                Tensor y_tensor = vector_of(y);

                SushiBLAS::BLASOps(recorder()).hemv(A, x_tensor, y_tensor, uplo,
                                                 std::complex<double>(1.0, 1.0), 2.0);
                recording->run().value();

                expect_near(read_row_ordered<T>(y_tensor), expected, K_HAND_ROUNDING);
            });
        }

        TEST_P(Medium_BlasComplexTriangle, HerUpdatesOneTriangleAndLeavesARealDiagonal)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const auto z = &make<T>;
                const std::vector<T> x = {z(1, 1), z(2, -1), z(0, 1)};
                const std::vector<T> ones(9, z(1, 0));
                const std::vector<T> updated = {z(5, 0), z(3, 6), z(3, -2),
                                                z(3, -6), z(11, 0), z(-1, -4),
                                                z(3, 2), z(-1, 4), z(3, 0)};

                Tensor x_tensor = vector_of(x);
                Tensor A = matrix_of(3, 3, with_imaginary_diagonal(keep_triangle(ones, 3, uplo), 3));

                SushiBLAS::BLASOps(recorder()).her(x_tensor, A, uplo, 2.0);
                recording->run().value();

                const std::vector<T> result = read_row_ordered<T>(A);
                expect_near(result, keep_triangle(updated, 3, uplo), K_HAND_ROUNDING);
                expect_real_diagonal(result, 3);
            });
        }

        TEST_P(Medium_BlasComplexTriangle, Her2UpdatesOneTriangleAndLeavesARealDiagonal)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const auto z = &make<T>;
                const std::vector<T> x = {z(1, 0), z(0, 1)};
                const std::vector<T> y = {z(1, 1), z(2, -1)};
                const std::vector<T> zeros(4);
                const std::vector<T> updated = {z(4, 0), z(1, 1),
                                                z(1, -1), z(-6, 0)};

                Tensor x_tensor = vector_of(x);
                Tensor y_tensor = vector_of(y);
                Tensor A = matrix_of(2, 2, with_imaginary_diagonal(keep_triangle(zeros, 2, uplo), 2));

                SushiBLAS::BLASOps(recorder()).her2(x_tensor, y_tensor, A, uplo,
                                                 std::complex<double>(1.0, 1.0));
                recording->run().value();

                const std::vector<T> result = read_row_ordered<T>(A);
                expect_near(result, keep_triangle(updated, 2, uplo), K_HAND_ROUNDING);
                expect_real_diagonal(result, 2);
            });
        }

        TEST_P(Medium_BlasComplexHerk, HerkUpdatesOneTriangleUntransposed)
        {
            const HerkForm form = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const auto z = &make<T>;
                const std::vector<T> a = {z(1, 0), z(0, 1), z(1, 1),
                                          z(2, 0), z(1, -1), z(0, -1)};
                const std::vector<T> prior = {z(1, 0), z(1, 1),
                                              z(1, -1), z(1, 0)};
                const std::vector<T> updated = {z(11, 0), z(3, 7),
                                                z(3, -7), z(17, 0)};

                Tensor A = matrix_of(2, 3, a);
                Tensor C = matrix_of(2, 2, with_imaginary_diagonal(keep_triangle(prior, 2, form.uplo), 2));

                SushiBLAS::BLASOps(recorder()).herk(A, C, form.uplo, Core::Transpose::NoTrans, 2.0, 3.0);
                recording->run().value();

                const std::vector<T> result = read_row_ordered<T>(C);
                expect_near(result, keep_triangle(updated, 2, form.uplo), K_HAND_ROUNDING);
                expect_real_diagonal(result, 2);
            });
        }

        TEST_P(Medium_BlasComplexHerk, HerkUpdatesOneTriangleConjugateTransposed)
        {
            const HerkForm form = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const auto z = &make<T>;
                const std::vector<T> a = {z(1, 0), z(2, 0),
                                          z(0, -1), z(1, 1),
                                          z(1, -1), z(0, 1)};
                const std::vector<T> prior = {z(1, 0), z(1, 1),
                                              z(1, -1), z(1, 0)};
                const std::vector<T> updated = {z(11, 0), z(3, 7),
                                                z(3, -7), z(17, 0)};

                Tensor A = matrix_of(3, 2, a);
                Tensor C = matrix_of(2, 2, with_imaginary_diagonal(keep_triangle(prior, 2, form.uplo), 2));

                SushiBLAS::BLASOps(recorder())
                    .herk(A, C, form.uplo, Core::Transpose::ConjTrans, 2.0, 3.0);
                recording->run().value();

                const std::vector<T> result = read_row_ordered<T>(C);
                expect_near(result, keep_triangle(updated, 2, form.uplo), K_HAND_ROUNDING);
                expect_real_diagonal(result, 2);
            });
        }

        TEST_P(Medium_BlasComplexPair, GemmConjTransEqualsTransOnAConjugatedCopy)
        {
            const TransposePair pair = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_calls_agree<T>(
                    [&] { return run_gemm<T>(as_given(pair.a), as_given(pair.b), {}); },
                    [&]
                    {
                        return run_gemm<T>(on_conjugated_copy(pair.a), on_conjugated_copy(pair.b), {});
                    });
            });
        }

        TEST_P(Medium_BlasComplexCopy, BatchedGemmConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_calls_agree<T>(
                    [&] { return run_gemm<T>(as_given(conj), as_given(conj), {K_CASE_BATCH}); },
                    [&]
                    {
                        return run_gemm<T>(on_conjugated_copy(conj), on_conjugated_copy(conj),
                                           {K_CASE_BATCH});
                    });
            });
        }

        TEST_P(Medium_BlasComplexCopy, GemvConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_calls_agree<T>([&] { return run_gemv<T>(as_given(conj)); },
                                      [&] { return run_gemv<T>(on_conjugated_copy(conj)); });
            });
        }

        TEST_P(Medium_BlasComplexCopyTriangle, TrmvConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Uplo uplo = GetParam().axis;
            const Core::Transpose conj = Core::Transpose::ConjTrans;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_calls_agree<T>([&] { return run_trmv<T>(uplo, as_given(conj)); },
                                      [&] { return run_trmv<T>(uplo, on_conjugated_copy(conj)); });
            });
        }

        TEST_P(Medium_BlasComplexCopyTriangle, TrsvConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Uplo uplo = GetParam().axis;
            const Core::Transpose conj = Core::Transpose::ConjTrans;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_calls_agree<T>([&] { return run_trsv<T>(uplo, as_given(conj)); },
                                      [&] { return run_trsv<T>(uplo, on_conjugated_copy(conj)); });
            });
        }

        TEST_P(Medium_BlasComplexCopySide, TrsmConjTransEqualsTransOnAConjugatedCopy)
        {
            const SideUplo form = GetParam().axis;
            const Core::Transpose conj = Core::Transpose::ConjTrans;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_calls_agree<T>(
                    [&] { return run_trsm<T>(form.side, form.uplo, as_given(conj)); },
                    [&] { return run_trsm<T>(form.side, form.uplo, on_conjugated_copy(conj)); });
            });
        }

        TEST_P(Medium_BlasComplexAgree, DotuMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>([&] { return run_dot<T>(&Level1::dotu); });
            });
        }

        TEST_P(Medium_BlasComplexAgree, DotcMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>([&] { return run_dot<T>(&Level1::dotc); });
            });
        }

        TEST_P(Medium_BlasComplexAgree, GeruMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const BLASScalar scale = scalar(alpha<T>());
                expect_modes_agree<T>([&] { return run_ger<T>(&Level2::geru, scale); });
            });
        }

        TEST_P(Medium_BlasComplexAgree, GercMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const BLASScalar scale = scalar(alpha<T>());
                expect_modes_agree<T>([&] { return run_ger<T>(&Level2::gerc, scale); });
            });
        }

        TEST_P(Medium_BlasComplexAgree, GemmConjTransOverNoTransMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&]
                    {
                        return run_gemm<T>(as_given(Core::Transpose::ConjTrans),
                                           as_given(Core::Transpose::NoTrans), {});
                    });
            });
        }

        TEST_P(Medium_BlasComplexAgree, GemmNoTransOverConjTransMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&]
                    {
                        return run_gemm<T>(as_given(Core::Transpose::NoTrans),
                                           as_given(Core::Transpose::ConjTrans), {});
                    });
            });
        }

        TEST_P(Medium_BlasComplexAgree, BatchedGemmConjTransMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&]
                    {
                        return run_gemm<T>(as_given(Core::Transpose::ConjTrans),
                                           as_given(Core::Transpose::ConjTrans), {K_CASE_BATCH});
                    });
            });
        }

        TEST_P(Medium_BlasComplexAgree, GemvConjTransMatchesAcrossModes)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&] { return run_gemv<T>(as_given(Core::Transpose::ConjTrans)); });
            });
        }

        TEST_P(Medium_BlasComplexAgreeTriangle, HemvMatchesAcrossModes)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>([&] { return run_hemv<T>(&Level2::hemv, uplo); });
            });
        }

        TEST_P(Medium_BlasComplexAgreeTriangle, HerMatchesAcrossModes)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>([&] { return run_her<T>(&Level2::her, uplo); });
            });
        }

        TEST_P(Medium_BlasComplexAgreeTriangle, Her2MatchesAcrossModes)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>([&] { return run_her2<T>(&Level2::her2, uplo); });
            });
        }

        TEST_P(Medium_BlasComplexAgreeTriangle, TrmvConjTransMatchesAcrossModes)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&] { return run_trmv<T>(uplo, as_given(Core::Transpose::ConjTrans)); });
            });
        }

        TEST_P(Medium_BlasComplexAgreeTriangle, TrsvConjTransMatchesAcrossModes)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&] { return run_trsv<T>(uplo, as_given(Core::Transpose::ConjTrans)); });
            });
        }

        TEST_P(Medium_BlasComplexAgreeHerk, HerkMatchesAcrossModes)
        {
            const HerkForm form = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&] { return run_herk<T>(&Level3::herk, form.uplo, form.trans); });
            });
        }

        TEST_P(Medium_BlasComplexAgreeSide, TrsmConjTransMatchesAcrossModes)
        {
            const SideUplo form = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                expect_modes_agree<T>(
                    [&]
                    {
                        return run_trsm<T>(form.side, form.uplo,
                                           as_given(Core::Transpose::ConjTrans));
                    });
            });
        }

        TEST_P(Medium_BlasRealForwardsTriangle, HemvGivesTheBitsOfSymv)
        {
            const Core::Uplo uplo = GetParam().axis;
            expect_calls_give_the_same_bits<float>(
                [&] { return run_hemv<float>(&Level2::hemv, uplo); },
                [&] { return run_hemv<float>(&Level2::symv, uplo); });
        }

        TEST_P(Medium_BlasRealForwardsTriangle, HerGivesTheBitsOfSyr)
        {
            const Core::Uplo uplo = GetParam().axis;
            expect_calls_give_the_same_bits<float>(
                [&] { return run_her<float>(&Level2::her, uplo); },
                [&] { return run_her<float>(&Level2::syr, uplo); });
        }

        TEST_P(Medium_BlasRealForwardsTriangle, Her2GivesTheBitsOfSyr2)
        {
            const Core::Uplo uplo = GetParam().axis;
            expect_calls_give_the_same_bits<float>(
                [&] { return run_her2<float>(&Level2::her2, uplo); },
                [&] { return run_her2<float>(&Level2::syr2, uplo); });
        }

        TEST_P(Medium_BlasRealForwardsTriangle, HerkUnderConjTransGivesTheBitsOfSyrkUnderTrans)
        {
            const Core::Uplo uplo = GetParam().axis;
            expect_calls_give_the_same_bits<float>(
                [&] { return run_herk<float>(&Level3::herk, uplo, Core::Transpose::ConjTrans); },
                [&] { return run_herk<float>(&Level3::syrk, uplo, Core::Transpose::Trans); });
        }

        TEST_P(Medium_BlasRealForwardsHerk, HerkGivesTheBitsOfSyrk)
        {
            const HerkForm form = GetParam().axis;
            expect_calls_give_the_same_bits<float>(
                [&] { return run_herk<float>(&Level3::herk, form.uplo, form.trans); },
                [&] { return run_herk<float>(&Level3::syrk, form.uplo, form.trans); });
        }

        TEST_P(Medium_BlasRealForwards, DotuGivesTheBitsOfDot)
        {
            expect_calls_give_the_same_bits<float>([&] { return run_dot<float>(&Level1::dotu); },
                                                   [&] { return run_dot<float>(&Level1::dot); });
        }

        TEST_P(Medium_BlasRealForwards, DotcGivesTheBitsOfDot)
        {
            expect_calls_give_the_same_bits<float>([&] { return run_dot<float>(&Level1::dotc); },
                                                   [&] { return run_dot<float>(&Level1::dot); });
        }

        TEST_P(Medium_BlasRealForwards, GeruGivesTheBitsOfGer)
        {
            expect_calls_give_the_same_bits<float>(
                [&] { return run_ger<float>(&Level2::geru, BLASScalar(K_REAL_ALPHA)); },
                [&] { return run_ger<float>(&Level2::ger, K_REAL_ALPHA); });
        }

        TEST_P(Medium_BlasRealForwards, GercGivesTheBitsOfGer)
        {
            expect_calls_give_the_same_bits<float>(
                [&] { return run_ger<float>(&Level2::gerc, BLASScalar(K_REAL_ALPHA)); },
                [&] { return run_ger<float>(&Level2::ger, K_REAL_ALPHA); });
        }

        TEST_P(Medium_BlasRealForwards, GemmConjTransOverNoTransGivesTheBitsOfTrans)
        {
            const OperandForm conj = as_given(Core::Transpose::ConjTrans);
            const OperandForm trans = as_given(Core::Transpose::Trans);
            const OperandForm none = as_given(Core::Transpose::NoTrans);

            expect_calls_give_the_same_bits<float>([&] { return run_gemm<float>(conj, none, {}); },
                                                   [&] { return run_gemm<float>(trans, none, {}); });
        }

        TEST_P(Medium_BlasRealForwards, GemmNoTransOverConjTransGivesTheBitsOfTrans)
        {
            const OperandForm conj = as_given(Core::Transpose::ConjTrans);
            const OperandForm trans = as_given(Core::Transpose::Trans);
            const OperandForm none = as_given(Core::Transpose::NoTrans);

            expect_calls_give_the_same_bits<float>([&] { return run_gemm<float>(none, conj, {}); },
                                                   [&] { return run_gemm<float>(none, trans, {}); });
        }

        TEST_P(Medium_BlasRealForwards, BatchedGemmConjTransGivesTheBitsOfTrans)
        {
            const OperandForm conj = as_given(Core::Transpose::ConjTrans);
            const OperandForm trans = as_given(Core::Transpose::Trans);

            expect_calls_give_the_same_bits<float>(
                [&] { return run_gemm<float>(conj, conj, {K_CASE_BATCH}); },
                [&] { return run_gemm<float>(trans, trans, {K_CASE_BATCH}); });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexTriangle,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_REFERENCE_MODE, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexHerk,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_REFERENCE_MODE,
                                        herk_forms(Core::Transpose::NoTrans,
                                                   Core::Transpose::ConjTrans))),
            row_name<HerkForm>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexPair,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES, K_GEMM_PAIRS)),
            row_name<TransposePair>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexCopy,
            ::testing::ValuesIn(plain_rows(K_COMPLEX_TYPES, K_BOTH_MODES)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexCopyTriangle,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexCopySide,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES, all_sides_and_triangles())),
            row_name<SideUplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexAgree,
            ::testing::ValuesIn(plain_rows(K_COMPLEX_TYPES, K_DEVICE_MODE)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexAgreeTriangle,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_DEVICE_MODE, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexAgreeHerk,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_DEVICE_MODE,
                                        herk_forms(Core::Transpose::NoTrans,
                                                   Core::Transpose::ConjTrans))),
            row_name<HerkForm>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasComplexAgreeSide,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_DEVICE_MODE, all_sides_and_triangles())),
            row_name<SideUplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasRealForwards,
            ::testing::ValuesIn(plain_rows(K_REAL_TYPE, K_BOTH_MODES)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasRealForwardsTriangle,
            ::testing::ValuesIn(rows_of(K_REAL_TYPE, K_BOTH_MODES, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasRealForwardsHerk,
            ::testing::ValuesIn(rows_of(K_REAL_TYPE, K_BOTH_MODES,
                                        herk_forms(Core::Transpose::NoTrans,
                                                   Core::Transpose::Trans))),
            row_name<HerkForm>);
    } // namespace Test
} // namespace SushiBLAS
