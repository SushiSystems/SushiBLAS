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
 * @file test_blas_complex.cpp
 * @brief Tests the conjugating BLAS routines, CONJ_TRANS and their forwards on a real tensor.
 * @author Mustafa Garip
 */

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <random>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Seed of the data every random case draws.
        constexpr std::uint32_t K_CONJUGATION_SEED = 20261007u;

        /// @brief Absolute tolerance of a hand-computed case; its values are small integers.
        constexpr double K_HAND_TOLERANCE = 1e-5;

        /// @brief Absolute tolerance between two results on binary32 parts of magnitude near 10.
        constexpr double K_FLOAT_PARITY_TOLERANCE = 1e-4;

        /// @brief Absolute tolerance between two results on binary64 parts of magnitude near 10.
        constexpr double K_DOUBLE_PARITY_TOLERANCE = 1e-11;

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

        /// @brief Both element orders a tensor can have.
        constexpr std::array<Core::Layout, 2> K_CASE_LAYOUTS = {Core::Layout::RowMajor,
                                                                Core::Layout::ColumnMajor};

        /// @brief Both kernels an operation has.
        constexpr std::array<ExecutionMode, 2> K_CASE_MODES = {ExecutionMode::ForceReference,
                                                               ExecutionMode::ForceSycl};

        /// @brief Both triangles a symmetric, Hermitian or triangular call can name.
        constexpr std::array<Core::Uplo, 2> K_CASE_TRIANGLES = {Core::Uplo::Upper,
                                                                Core::Uplo::Lower};

        /// @brief Both sides a triangular matrix can stand on.
        constexpr std::array<Core::Side, 2> K_CASE_SIDES = {Core::Side::Left, Core::Side::Right};

        /**
         * @brief Builds the operands of a BLAS call from seeded data, records the call and reads
         *        its result in row order, whatever the layout of the engine.
         * @tparam T The element type of every operand.
         */
        template<typename T>
        class BlasConjugationCases : public SushiBLASTest
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

                /** @brief Rebuilds the engine in layout and mode, and restarts the data. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    reinit_engine(layout);
                    engine->set_mode(mode);
                    restart_data();
                }

                /** @brief Makes the next draw repeat the first one. */
                void restart_data()
                {
                    generator_.seed(K_CONJUGATION_SEED);
                }

                /** @brief Returns real + imaginary * i in T; a real T drops the imaginary part. */
                [[nodiscard]] static T make(double real, [[maybe_unused]] double imaginary) noexcept
                {
                    using R = Kernels::RealType<T>;
                    if constexpr (Core::is_complex_v<T>)
                        return T(static_cast<R>(real), static_cast<R>(imaginary));
                    else
                        return static_cast<R>(real);
                }

                /** @brief Returns value as the scalar argument of a call. */
                [[nodiscard]] static BlasScalar scalar(const T& value) noexcept
                {
                    if constexpr (Core::is_complex_v<T>)
                        return BlasScalar(std::complex<double>(static_cast<double>(value.real()),
                                                               static_cast<double>(value.imag())));
                    else
                        return BlasScalar(static_cast<double>(value));
                }

                /** @brief Returns the scalar that multiplies the product or the update. */
                [[nodiscard]] static T alpha() noexcept
                {
                    return make(1.5, -0.5);
                }

                /** @brief Returns the scalar that multiplies the prior result. */
                [[nodiscard]] static T beta() noexcept
                {
                    return make(0.25, 0.75);
                }

                /** @brief Returns the tolerance between two results in the component type of T. */
                [[nodiscard]] static constexpr double parity_tolerance() noexcept
                {
                    return std::is_same_v<Kernels::RealType<T>, float> ? K_FLOAT_PARITY_TOLERANCE
                                                                       : K_DOUBLE_PARITY_TOLERANCE;
                }

                /** @brief Returns count values with every part drawn from [-1, 1]. */
                [[nodiscard]] std::vector<T> random_values(int64_t count)
                {
                    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                    std::vector<T> values(static_cast<std::size_t>(count));
                    for (T& value : values)
                    {
                        const double real = distribution(generator_);
                        const double imaginary = distribution(generator_);
                        value = make(real, imaginary);
                    }
                    return values;
                }

                /** @brief Returns random_values, conjugated when form asks for it. */
                [[nodiscard]] std::vector<T> operand_values(int64_t count, const OperandForm& form)
                {
                    std::vector<T> values = random_values(count);
                    if (form.conjugate_values)
                        for (T& value : values)
                            value = Kernels::conjugate(value);
                    return values;
                }

                /** @brief Returns an n x n operand in row order whose diagonal is shifted. */
                [[nodiscard]] std::vector<T> triangular_values(int64_t n, const OperandForm& form)
                {
                    std::vector<T> values = operand_values(n * n, form);
                    for (int64_t i = 0; i < n; ++i)
                        values[static_cast<std::size_t>(i * n + i)] += make(K_DIAGONAL_SHIFT, 0.0);
                    return values;
                }

                /** @brief Returns a rank-1 tensor holding values. */
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    return tensor_of({static_cast<int64_t>(values.size())}, values);
                }

                /** @brief Returns a rows x cols tensor holding the row-ordered values. */
                [[nodiscard]] Tensor matrix_of(int64_t rows, int64_t cols,
                                               const std::vector<T>& row_ordered)
                {
                    return matrices_of({}, rows, cols, row_ordered);
                }

                /**
                 * @brief Returns a tensor of batch_axes x rows x cols holding the row-ordered
                 *        matrices, stored in the layout of the engine.
                 */
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
                template<typename Call>
                [[nodiscard]] std::vector<T> run_dot(Call call)
                {
                    Tensor x = vector_of(random_values(K_CASE_ORDER));
                    Tensor y = vector_of(random_values(K_CASE_ORDER));
                    Tensor result = vector_of(std::vector<T>(1));

                    SushiBLAS::BLASOps ops(*engine);
                    (ops.*call)(x, y, result);
                    engine->execute();
                    return read_row_ordered(result);
                }

                /** @brief Returns A after a rank-1 update call scaled by scale. */
                template<typename Call, typename Scalar>
                [[nodiscard]] std::vector<T> run_ger(Call call, Scalar scale)
                {
                    Tensor x = vector_of(random_values(K_CASE_ROWS));
                    Tensor y = vector_of(random_values(K_CASE_COLS));
                    Tensor A = matrix_of(K_CASE_ROWS, K_CASE_COLS,
                                         random_values(K_CASE_ROWS * K_CASE_COLS));

                    SushiBLAS::BLASOps ops(*engine);
                    (ops.*call)(x, y, A, scale);
                    engine->execute();
                    return read_row_ordered(A);
                }

                /** @brief Returns y after a symmetric or Hermitian matrix-vector call. */
                template<typename Call>
                [[nodiscard]] std::vector<T> run_hemv(Call call, Core::Uplo uplo)
                {
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values(K_CASE_ORDER * K_CASE_ORDER));
                    Tensor x = vector_of(random_values(K_CASE_ORDER));
                    Tensor y = vector_of(random_values(K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(*engine);
                    (ops.*call)(A, x, y, uplo, scalar(alpha()), scalar(beta()));
                    engine->execute();
                    return read_row_ordered(y);
                }

                /** @brief Returns A after a symmetric or Hermitian rank-1 update call. */
                template<typename Call>
                [[nodiscard]] std::vector<T> run_her(Call call, Core::Uplo uplo)
                {
                    Tensor x = vector_of(random_values(K_CASE_ORDER));
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values(K_CASE_ORDER * K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(*engine);
                    (ops.*call)(x, A, uplo, K_REAL_ALPHA);
                    engine->execute();
                    return read_row_ordered(A);
                }

                /** @brief Returns A after a symmetric or Hermitian rank-2 update call. */
                template<typename Call>
                [[nodiscard]] std::vector<T> run_her2(Call call, Core::Uplo uplo)
                {
                    Tensor x = vector_of(random_values(K_CASE_ORDER));
                    Tensor y = vector_of(random_values(K_CASE_ORDER));
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values(K_CASE_ORDER * K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(*engine);
                    (ops.*call)(x, y, A, uplo, scalar(alpha()));
                    engine->execute();
                    return read_row_ordered(A);
                }

                /** @brief Returns C after a symmetric or Hermitian rank-k update call. */
                template<typename Call>
                [[nodiscard]] std::vector<T> run_herk(Call call, Core::Uplo uplo,
                                                      Core::Transpose trans)
                {
                    const bool transposed = Core::is_transposed(trans);
                    const int64_t a_rows = transposed ? K_CASE_INNER : K_CASE_ORDER;
                    const int64_t a_cols = transposed ? K_CASE_ORDER : K_CASE_INNER;
                    Tensor A = matrix_of(a_rows, a_cols, random_values(a_rows * a_cols));
                    Tensor C = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         random_values(K_CASE_ORDER * K_CASE_ORDER));

                    SushiBLAS::BLASOps ops(*engine);
                    (ops.*call)(A, C, uplo, trans, K_REAL_ALPHA, K_REAL_BETA);
                    engine->execute();
                    return read_row_ordered(C);
                }

                /** @brief Returns C = alpha * op(A) * op(B) + beta * C over drawn operands. */
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
                                           operand_values(batch * a_rows * a_cols, a));
                    Tensor B = matrices_of(batch_axes, b_rows, b_cols,
                                           operand_values(batch * b_rows * b_cols, b));
                    Tensor C = matrices_of(batch_axes, K_CASE_ROWS, K_CASE_COLS,
                                           random_values(batch * K_CASE_ROWS * K_CASE_COLS));

                    SushiBLAS::BLASOps(*engine).gemm(A, B, C, a.trans, b.trans, scalar(alpha()),
                                                     scalar(beta()));
                    engine->execute();
                    return read_row_ordered(C);
                }

                /** @brief Returns y = alpha * op(A) * x + beta * y over drawn operands. */
                [[nodiscard]] std::vector<T> run_gemv(const OperandForm& a)
                {
                    const bool transposed = Core::is_transposed(a.trans);
                    Tensor A = matrix_of(K_CASE_ROWS, K_CASE_COLS,
                                         operand_values(K_CASE_ROWS * K_CASE_COLS, a));
                    Tensor x = vector_of(random_values(transposed ? K_CASE_ROWS : K_CASE_COLS));
                    Tensor y = vector_of(random_values(transposed ? K_CASE_COLS : K_CASE_ROWS));

                    SushiBLAS::BLASOps(*engine).gemv(A, x, y, a.trans, scalar(alpha()),
                                                     scalar(beta()));
                    engine->execute();
                    return read_row_ordered(y);
                }

                /** @brief Returns x = op(A) * x for a drawn triangular A. */
                [[nodiscard]] std::vector<T> run_trmv(Core::Uplo uplo, const OperandForm& a)
                {
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         triangular_values(K_CASE_ORDER, a));
                    Tensor x = vector_of(random_values(K_CASE_ORDER));

                    SushiBLAS::BLASOps(*engine).trmv(A, x, uplo, a.trans, Core::Diag::NonUnit);
                    engine->execute();
                    return read_row_ordered(x);
                }

                /** @brief Returns the solution of op(A) * x = b for a drawn triangular A. */
                [[nodiscard]] std::vector<T> run_trsv(Core::Uplo uplo, const OperandForm& a)
                {
                    Tensor A = matrix_of(K_CASE_ORDER, K_CASE_ORDER,
                                         triangular_values(K_CASE_ORDER, a));
                    Tensor b = vector_of(random_values(K_CASE_ORDER));

                    SushiBLAS::BLASOps(*engine).trsv(A, b, uplo, a.trans, Core::Diag::NonUnit);
                    engine->execute();
                    return read_row_ordered(b);
                }

                /** @brief Returns the solution of a triangular system with a drawn A on side. */
                [[nodiscard]] std::vector<T> run_trsm(Core::Side side, Core::Uplo uplo,
                                                      const OperandForm& a)
                {
                    const int64_t order = (side == Core::Side::Left) ? K_CASE_ROWS : K_CASE_COLS;
                    Tensor A = matrix_of(order, order, triangular_values(order, a));
                    Tensor B = matrix_of(K_CASE_ROWS, K_CASE_COLS,
                                         random_values(K_CASE_ROWS * K_CASE_COLS));

                    SushiBLAS::BLASOps(*engine).trsm(A, B, side, uplo, a.trans,
                                                     Core::Diag::NonUnit, scalar(alpha()));
                    engine->execute();
                    return read_row_ordered(B);
                }

                /** @brief Expects every part of every element within tolerance of expected. */
                static void expect_near(const std::vector<T>& actual,
                                        const std::vector<T>& expected, double tolerance)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const std::complex<double> a = Detail::to_complex_double(actual[i]);
                        const std::complex<double> e = Detail::to_complex_double(expected[i]);
                        EXPECT_NEAR(a.real(), e.real(), tolerance) << "real part of element " << i;
                        EXPECT_NEAR(a.imag(), e.imag(), tolerance)
                            << "imaginary part of element " << i;
                    }
                }

                /** @brief Expects two host vectors to hold the same bytes, element by element. */
                static void expect_same_bits(const std::vector<T>& actual,
                                             const std::vector<T>& expected)
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

                /** @brief Expects call to give the same result on both kernels. */
                template<typename Call>
                void expect_modes_agree(Call&& call)
                {
                    for (const Core::Layout layout : K_CASE_LAYOUTS)
                    {
                        start(layout, ExecutionMode::ForceReference);
                        const std::vector<T> reference = call();
                        start(layout, ExecutionMode::ForceSycl);
                        const std::vector<T> device = call();

                        expect_near(device, reference, parity_tolerance());
                    }
                }

                /** @brief Expects two calls over the same draws to agree in every engine. */
                template<typename First, typename Second>
                void expect_calls_agree(First&& first, Second&& second)
                {
                    for (const Core::Layout layout : K_CASE_LAYOUTS)
                    {
                        for (const ExecutionMode mode : K_CASE_MODES)
                        {
                            start(layout, mode);
                            const std::vector<T> from_first = first();
                            restart_data();
                            const std::vector<T> from_second = second();

                            expect_near(from_first, from_second, parity_tolerance());
                        }
                    }
                }

                /** @brief Expects two calls over the same draws to give the same bits. */
                template<typename First, typename Second>
                void expect_calls_give_the_same_bits(First&& first, Second&& second)
                {
                    for (const Core::Layout layout : K_CASE_LAYOUTS)
                    {
                        for (const ExecutionMode mode : K_CASE_MODES)
                        {
                            start(layout, mode);
                            const std::vector<T> from_first = first();
                            restart_data();
                            const std::vector<T> from_second = second();

                            expect_same_bits(from_first, from_second);
                        }
                    }
                }

            private:
                /** @brief Returns where element (r, c) of a rows x cols matrix is stored. */
                [[nodiscard]] std::size_t storage_offset(int64_t r, int64_t c, int64_t rows,
                                                         int64_t cols) const
                {
                    return static_cast<std::size_t>(
                        (engine->get_layout() == Core::Layout::RowMajor) ? r * cols + c
                                                                          : r + c * rows);
                }

                /** @brief Returns a tensor of dims holding stored in storage order. */
                [[nodiscard]] Tensor tensor_of(const std::vector<int64_t>& dims,
                                               const std::vector<T>& stored)
                {
                    Tensor tensor = engine->create_tensor(
                        SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                        get_sushiblas_dtype<T>());
                    TestEnvironment::get_context().devices().queue(0)
                        .memcpy(tensor.data(), stored.data(), stored.size() * sizeof(T)).wait();
                    return tensor;
                }

                /// @brief The generator every draw advances.
                std::mt19937 generator_{K_CONJUGATION_SEED};
        };

        /**
         * @brief Runs the conjugating routines and CONJ_TRANS on a complex element type.
         */
        template<typename T>
        class Unit_BlasComplex : public BlasConjugationCases<T>
        {
            protected:
                /** @brief Returns real + imaginary * i in T. */
                [[nodiscard]] static T z(double real, double imaginary) noexcept
                {
                    return BlasConjugationCases<T>::make(real, imaginary);
                }

                /** @brief Returns full with the triangle uplo does not name poisoned. */
                [[nodiscard]] static std::vector<T> keep_triangle(std::vector<T> full, int64_t n,
                                                                  Core::Uplo uplo)
                {
                    const bool upper = (uplo == Core::Uplo::Upper);
                    for (int64_t r = 0; r < n; ++r)
                    {
                        for (int64_t c = 0; c < n; ++c)
                        {
                            const bool kept = upper ? (r <= c) : (r >= c);
                            if (!kept)
                                full[static_cast<std::size_t>(r * n + c)] = z(K_POISON, K_POISON);
                        }
                    }
                    return full;
                }

                /** @brief Returns matrix with K_DIAGONAL_GARBAGE on its imaginary diagonal. */
                [[nodiscard]] static std::vector<T> with_imaginary_diagonal(std::vector<T> matrix,
                                                                            int64_t n)
                {
                    for (int64_t i = 0; i < n; ++i)
                        matrix[static_cast<std::size_t>(i * n + i)] += z(0.0, K_DIAGONAL_GARBAGE);
                    return matrix;
                }

                /** @brief Expects the diagonal of an n x n matrix to be exactly real. */
                static void expect_real_diagonal(const std::vector<T>& matrix, int64_t n)
                {
                    for (int64_t i = 0; i < n; ++i)
                        EXPECT_EQ(matrix[static_cast<std::size_t>(i * n + i)].imag(), 0)
                            << "diagonal element " << i;
                }
        };

        using BlasComplexTypes = ::testing::Types<std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_BlasComplex, BlasComplexTypes);

        TYPED_TEST(Unit_BlasComplex, DotuSumsTheUnconjugatedProducts)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> x = {z(1, 2), z(3, -1)};
            const std::vector<T> y = {z(2, -1), z(0, 1)};
            const std::vector<T> expected = {z(5, 6)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                this->start(layout, ExecutionMode::ForceReference);
                Tensor x_tensor = this->vector_of(x);
                Tensor y_tensor = this->vector_of(y);
                Tensor result = this->vector_of(std::vector<T>(1));

                SushiBLAS::BLASOps(*this->engine).dotu(x_tensor, y_tensor, result);
                this->engine->execute();

                TestFixture::expect_near(this->read_row_ordered(result), expected,
                                         K_HAND_TOLERANCE);
            }
        }

        TYPED_TEST(Unit_BlasComplex, DotcConjugatesTheFirstVector)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> x = {z(1, 2), z(3, -1)};
            const std::vector<T> y = {z(2, -1), z(0, 1)};
            const std::vector<T> expected = {z(-1, -2)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                this->start(layout, ExecutionMode::ForceReference);
                Tensor x_tensor = this->vector_of(x);
                Tensor y_tensor = this->vector_of(y);
                Tensor result = this->vector_of(std::vector<T>(1));

                SushiBLAS::BLASOps(*this->engine).dotc(x_tensor, y_tensor, result);
                this->engine->execute();

                TestFixture::expect_near(this->read_row_ordered(result), expected,
                                         K_HAND_TOLERANCE);
            }
        }

        TYPED_TEST(Unit_BlasComplex, GeruAddsTheUnconjugatedOuterProduct)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> x = {z(1, 1), z(2, 0)};
            const std::vector<T> y = {z(1, 0), z(0, 1), z(1, -1)};
            const std::vector<T> expected = {z(0, 1), z(0, -1), z(1, 2),
                                             z(1, 2), z(-1, 0), z(3, 2)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                this->start(layout, ExecutionMode::ForceReference);
                Tensor x_tensor = this->vector_of(x);
                Tensor y_tensor = this->vector_of(y);
                Tensor A = this->matrix_of(2, 3, std::vector<T>(6, z(1, 0)));

                SushiBLAS::BLASOps(*this->engine).geru(x_tensor, y_tensor, A,
                                                       std::complex<double>(0.0, 1.0));
                this->engine->execute();

                TestFixture::expect_near(this->read_row_ordered(A), expected, K_HAND_TOLERANCE);
            }
        }

        TYPED_TEST(Unit_BlasComplex, GercConjugatesTheSecondVector)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> x = {z(1, 1), z(2, 0)};
            const std::vector<T> y = {z(1, 0), z(0, 1), z(1, -1)};
            const std::vector<T> expected = {z(0, 1), z(2, 1), z(-1, 0),
                                             z(1, 2), z(3, 0), z(-1, 2)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                this->start(layout, ExecutionMode::ForceReference);
                Tensor x_tensor = this->vector_of(x);
                Tensor y_tensor = this->vector_of(y);
                Tensor A = this->matrix_of(2, 3, std::vector<T>(6, z(1, 0)));

                SushiBLAS::BLASOps(*this->engine).gerc(x_tensor, y_tensor, A,
                                                       std::complex<double>(0.0, 1.0));
                this->engine->execute();

                TestFixture::expect_near(this->read_row_ordered(A), expected, K_HAND_TOLERANCE);
            }
        }

        TYPED_TEST(Unit_BlasComplex, HemvReadsOneTriangleAndNoImaginaryDiagonal)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> hermitian = {z(2, 0), z(1, 1), z(0, -1),
                                              z(1, -1), z(3, 0), z(2, 1),
                                              z(0, 1), z(2, -1), z(-1, 0)};
            const std::vector<T> x = {z(1, 0), z(0, 1), z(2, 0)};
            const std::vector<T> expected = {z(1, -1), z(5, 4), z(-1, 3)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                {
                    this->start(layout, ExecutionMode::ForceReference);
                    Tensor A = this->matrix_of(3, 3, TestFixture::with_imaginary_diagonal(
                        TestFixture::keep_triangle(hermitian, 3, uplo), 3));
                    Tensor x_tensor = this->vector_of(x);
                    Tensor y_tensor = this->vector_of(std::vector<T>(3));

                    SushiBLAS::BLASOps(*this->engine).hemv(A, x_tensor, y_tensor, uplo, 1.0, 0.0);
                    this->engine->execute();

                    TestFixture::expect_near(this->read_row_ordered(y_tensor), expected,
                                             K_HAND_TOLERANCE);
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, HemvScalesAndAccumulates)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> hermitian = {z(2, 0), z(1, 1), z(0, -1),
                                              z(1, -1), z(3, 0), z(2, 1),
                                              z(0, 1), z(2, -1), z(-1, 0)};
            const std::vector<T> x = {z(1, 0), z(0, 1), z(2, 0)};
            const std::vector<T> y = {z(1, 0), z(0, 1), z(-1, 0)};
            const std::vector<T> expected = {z(4, 0), z(1, 11), z(-6, 2)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                {
                    this->start(layout, ExecutionMode::ForceReference);
                    Tensor A = this->matrix_of(3, 3, TestFixture::with_imaginary_diagonal(
                        TestFixture::keep_triangle(hermitian, 3, uplo), 3));
                    Tensor x_tensor = this->vector_of(x);
                    Tensor y_tensor = this->vector_of(y);

                    SushiBLAS::BLASOps(*this->engine).hemv(A, x_tensor, y_tensor, uplo,
                                                           std::complex<double>(1.0, 1.0), 2.0);
                    this->engine->execute();

                    TestFixture::expect_near(this->read_row_ordered(y_tensor), expected,
                                             K_HAND_TOLERANCE);
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, HerUpdatesOneTriangleAndLeavesARealDiagonal)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> x = {z(1, 1), z(2, -1), z(0, 1)};
            const std::vector<T> ones(9, z(1, 0));
            const std::vector<T> updated = {z(5, 0), z(3, 6), z(3, -2),
                                            z(3, -6), z(11, 0), z(-1, -4),
                                            z(3, 2), z(-1, 4), z(3, 0)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                {
                    this->start(layout, ExecutionMode::ForceReference);
                    Tensor x_tensor = this->vector_of(x);
                    Tensor A = this->matrix_of(3, 3, TestFixture::with_imaginary_diagonal(
                        TestFixture::keep_triangle(ones, 3, uplo), 3));

                    SushiBLAS::BLASOps(*this->engine).her(x_tensor, A, uplo, 2.0);
                    this->engine->execute();

                    const std::vector<T> result = this->read_row_ordered(A);
                    TestFixture::expect_near(result, TestFixture::keep_triangle(updated, 3, uplo),
                                             K_HAND_TOLERANCE);
                    TestFixture::expect_real_diagonal(result, 3);
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, Her2UpdatesOneTriangleAndLeavesARealDiagonal)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> x = {z(1, 0), z(0, 1)};
            const std::vector<T> y = {z(1, 1), z(2, -1)};
            const std::vector<T> zeros(4);
            const std::vector<T> updated = {z(4, 0), z(1, 1),
                                            z(1, -1), z(-6, 0)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                {
                    this->start(layout, ExecutionMode::ForceReference);
                    Tensor x_tensor = this->vector_of(x);
                    Tensor y_tensor = this->vector_of(y);
                    Tensor A = this->matrix_of(2, 2, TestFixture::with_imaginary_diagonal(
                        TestFixture::keep_triangle(zeros, 2, uplo), 2));

                    SushiBLAS::BLASOps(*this->engine).her2(x_tensor, y_tensor, A, uplo,
                                                           std::complex<double>(1.0, 1.0));
                    this->engine->execute();

                    const std::vector<T> result = this->read_row_ordered(A);
                    TestFixture::expect_near(result, TestFixture::keep_triangle(updated, 2, uplo),
                                             K_HAND_TOLERANCE);
                    TestFixture::expect_real_diagonal(result, 2);
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, HerkUpdatesOneTriangleUntransposed)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> a = {z(1, 0), z(0, 1), z(1, 1),
                                      z(2, 0), z(1, -1), z(0, -1)};
            const std::vector<T> prior = {z(1, 0), z(1, 1),
                                          z(1, -1), z(1, 0)};
            const std::vector<T> updated = {z(11, 0), z(3, 7),
                                            z(3, -7), z(17, 0)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                {
                    this->start(layout, ExecutionMode::ForceReference);
                    Tensor A = this->matrix_of(2, 3, a);
                    Tensor C = this->matrix_of(2, 2, TestFixture::with_imaginary_diagonal(
                        TestFixture::keep_triangle(prior, 2, uplo), 2));

                    SushiBLAS::BLASOps(*this->engine).herk(A, C, uplo, Core::Transpose::NoTrans,
                                                           2.0, 3.0);
                    this->engine->execute();

                    const std::vector<T> result = this->read_row_ordered(C);
                    TestFixture::expect_near(result, TestFixture::keep_triangle(updated, 2, uplo),
                                             K_HAND_TOLERANCE);
                    TestFixture::expect_real_diagonal(result, 2);
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, HerkUpdatesOneTriangleConjugateTransposed)
        {
            using T = TypeParam;
            const auto z = &TestFixture::z;
            const std::vector<T> a = {z(1, 0), z(2, 0),
                                      z(0, -1), z(1, 1),
                                      z(1, -1), z(0, 1)};
            const std::vector<T> prior = {z(1, 0), z(1, 1),
                                          z(1, -1), z(1, 0)};
            const std::vector<T> updated = {z(11, 0), z(3, 7),
                                            z(3, -7), z(17, 0)};

            for (const Core::Layout layout : K_CASE_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                {
                    this->start(layout, ExecutionMode::ForceReference);
                    Tensor A = this->matrix_of(3, 2, a);
                    Tensor C = this->matrix_of(2, 2, TestFixture::with_imaginary_diagonal(
                        TestFixture::keep_triangle(prior, 2, uplo), 2));

                    SushiBLAS::BLASOps(*this->engine).herk(A, C, uplo, Core::Transpose::ConjTrans,
                                                           2.0, 3.0);
                    this->engine->execute();

                    const std::vector<T> result = this->read_row_ordered(C);
                    TestFixture::expect_near(result, TestFixture::keep_triangle(updated, 2, uplo),
                                             K_HAND_TOLERANCE);
                    TestFixture::expect_real_diagonal(result, 2);
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, GemmConjTransEqualsTransOnAConjugatedCopy)
        {
            using Pair = std::array<Core::Transpose, 2>;
            const std::array<Pair, 5> pairs = {
                Pair{Core::Transpose::ConjTrans, Core::Transpose::NoTrans},
                Pair{Core::Transpose::NoTrans, Core::Transpose::ConjTrans},
                Pair{Core::Transpose::ConjTrans, Core::Transpose::ConjTrans},
                Pair{Core::Transpose::ConjTrans, Core::Transpose::Trans},
                Pair{Core::Transpose::Trans, Core::Transpose::ConjTrans}};

            for (const Pair& pair : pairs)
            {
                this->expect_calls_agree(
                    [&]
                    {
                        return this->run_gemm(TestFixture::as_given(pair[0]),
                                              TestFixture::as_given(pair[1]), {});
                    },
                    [&]
                    {
                        return this->run_gemm(TestFixture::on_conjugated_copy(pair[0]),
                                              TestFixture::on_conjugated_copy(pair[1]), {});
                    });
            }
        }

        TYPED_TEST(Unit_BlasComplex, BatchedGemmConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            this->expect_calls_agree(
                [&]
                {
                    return this->run_gemm(TestFixture::as_given(conj), TestFixture::as_given(conj),
                                          {K_CASE_BATCH});
                },
                [&]
                {
                    return this->run_gemm(TestFixture::on_conjugated_copy(conj),
                                          TestFixture::on_conjugated_copy(conj), {K_CASE_BATCH});
                });
        }

        TYPED_TEST(Unit_BlasComplex, GemvConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            this->expect_calls_agree(
                [&] { return this->run_gemv(TestFixture::as_given(conj)); },
                [&] { return this->run_gemv(TestFixture::on_conjugated_copy(conj)); });
        }

        TYPED_TEST(Unit_BlasComplex, TrmvConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
            {
                this->expect_calls_agree(
                    [&] { return this->run_trmv(uplo, TestFixture::as_given(conj)); },
                    [&] { return this->run_trmv(uplo, TestFixture::on_conjugated_copy(conj)); });
            }
        }

        TYPED_TEST(Unit_BlasComplex, TrsvConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
            {
                this->expect_calls_agree(
                    [&] { return this->run_trsv(uplo, TestFixture::as_given(conj)); },
                    [&] { return this->run_trsv(uplo, TestFixture::on_conjugated_copy(conj)); });
            }
        }

        TYPED_TEST(Unit_BlasComplex, TrsmConjTransEqualsTransOnAConjugatedCopy)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            for (const Core::Side side : K_CASE_SIDES)
            {
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                {
                    this->expect_calls_agree(
                        [&] { return this->run_trsm(side, uplo, TestFixture::as_given(conj)); },
                        [&]
                        {
                            return this->run_trsm(side, uplo,
                                                  TestFixture::on_conjugated_copy(conj));
                        });
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, DotuAndDotcMatchAcrossModes)
        {
            this->expect_modes_agree([&] { return this->run_dot(&Level1::dotu); });
            this->expect_modes_agree([&] { return this->run_dot(&Level1::dotc); });
        }

        TYPED_TEST(Unit_BlasComplex, GeruAndGercMatchAcrossModes)
        {
            const BlasScalar alpha = TestFixture::scalar(TestFixture::alpha());

            this->expect_modes_agree([&] { return this->run_ger(&Level2::geru, alpha); });
            this->expect_modes_agree([&] { return this->run_ger(&Level2::gerc, alpha); });
        }

        TYPED_TEST(Unit_BlasComplex, HemvMatchesAcrossModes)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                this->expect_modes_agree([&] { return this->run_hemv(&Level2::hemv, uplo); });
        }

        TYPED_TEST(Unit_BlasComplex, HerMatchesAcrossModes)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                this->expect_modes_agree([&] { return this->run_her(&Level2::her, uplo); });
        }

        TYPED_TEST(Unit_BlasComplex, Her2MatchesAcrossModes)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                this->expect_modes_agree([&] { return this->run_her2(&Level2::her2, uplo); });
        }

        TYPED_TEST(Unit_BlasComplex, HerkMatchesAcrossModes)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
            {
                for (const Core::Transpose trans :
                     {Core::Transpose::NoTrans, Core::Transpose::ConjTrans})
                {
                    this->expect_modes_agree(
                        [&] { return this->run_herk(&Level3::herk, uplo, trans); });
                }
            }
        }

        TYPED_TEST(Unit_BlasComplex, GemmConjTransMatchesAcrossModes)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;
            const Core::Transpose none = Core::Transpose::NoTrans;

            this->expect_modes_agree(
                [&]
                {
                    return this->run_gemm(TestFixture::as_given(conj), TestFixture::as_given(none),
                                          {});
                });
            this->expect_modes_agree(
                [&]
                {
                    return this->run_gemm(TestFixture::as_given(none), TestFixture::as_given(conj),
                                          {});
                });
            this->expect_modes_agree(
                [&]
                {
                    return this->run_gemm(TestFixture::as_given(conj), TestFixture::as_given(conj),
                                          {K_CASE_BATCH});
                });
        }

        TYPED_TEST(Unit_BlasComplex, GemvConjTransMatchesAcrossModes)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            this->expect_modes_agree([&] { return this->run_gemv(TestFixture::as_given(conj)); });
        }

        TYPED_TEST(Unit_BlasComplex, TrmvConjTransMatchesAcrossModes)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                this->expect_modes_agree(
                    [&] { return this->run_trmv(uplo, TestFixture::as_given(conj)); });
        }

        TYPED_TEST(Unit_BlasComplex, TrsvConjTransMatchesAcrossModes)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                this->expect_modes_agree(
                    [&] { return this->run_trsv(uplo, TestFixture::as_given(conj)); });
        }

        TYPED_TEST(Unit_BlasComplex, TrsmConjTransMatchesAcrossModes)
        {
            const Core::Transpose conj = Core::Transpose::ConjTrans;

            for (const Core::Side side : K_CASE_SIDES)
                for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                    this->expect_modes_agree(
                        [&] { return this->run_trsm(side, uplo, TestFixture::as_given(conj)); });
        }

        /**
         * @brief Runs each conjugating routine and its unconjugated sibling on a FLOAT32 tensor.
         */
        class Unit_BlasRealForwards : public BlasConjugationCases<float>
        {

        };

        TEST_F(Unit_BlasRealForwards, HemvGivesTheBitsOfSymv)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                expect_calls_give_the_same_bits([&] { return run_hemv(&Level2::hemv, uplo); },
                                                [&] { return run_hemv(&Level2::symv, uplo); });
        }

        TEST_F(Unit_BlasRealForwards, HerGivesTheBitsOfSyr)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                expect_calls_give_the_same_bits([&] { return run_her(&Level2::her, uplo); },
                                                [&] { return run_her(&Level2::syr, uplo); });
        }

        TEST_F(Unit_BlasRealForwards, Her2GivesTheBitsOfSyr2)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                expect_calls_give_the_same_bits([&] { return run_her2(&Level2::her2, uplo); },
                                                [&] { return run_her2(&Level2::syr2, uplo); });
        }

        TEST_F(Unit_BlasRealForwards, HerkGivesTheBitsOfSyrk)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
            {
                for (const Core::Transpose trans :
                     {Core::Transpose::NoTrans, Core::Transpose::Trans})
                {
                    expect_calls_give_the_same_bits(
                        [&] { return run_herk(&Level3::herk, uplo, trans); },
                        [&] { return run_herk(&Level3::syrk, uplo, trans); });
                }
            }
        }

        TEST_F(Unit_BlasRealForwards, HerkUnderConjTransGivesTheBitsOfSyrkUnderTrans)
        {
            for (const Core::Uplo uplo : K_CASE_TRIANGLES)
                expect_calls_give_the_same_bits(
                    [&] { return run_herk(&Level3::herk, uplo, Core::Transpose::ConjTrans); },
                    [&] { return run_herk(&Level3::syrk, uplo, Core::Transpose::Trans); });
        }

        TEST_F(Unit_BlasRealForwards, DotuAndDotcGiveTheBitsOfDot)
        {
            expect_calls_give_the_same_bits([&] { return run_dot(&Level1::dotu); },
                                            [&] { return run_dot(&Level1::dot); });
            expect_calls_give_the_same_bits([&] { return run_dot(&Level1::dotc); },
                                            [&] { return run_dot(&Level1::dot); });
        }

        TEST_F(Unit_BlasRealForwards, GeruAndGercGiveTheBitsOfGer)
        {
            expect_calls_give_the_same_bits(
                [&] { return run_ger(&Level2::geru, BlasScalar(K_REAL_ALPHA)); },
                [&] { return run_ger(&Level2::ger, K_REAL_ALPHA); });
            expect_calls_give_the_same_bits(
                [&] { return run_ger(&Level2::gerc, BlasScalar(K_REAL_ALPHA)); },
                [&] { return run_ger(&Level2::ger, K_REAL_ALPHA); });
        }

        TEST_F(Unit_BlasRealForwards, GemmUnderConjTransGivesTheBitsOfGemmUnderTrans)
        {
            const OperandForm conj = as_given(Core::Transpose::ConjTrans);
            const OperandForm trans = as_given(Core::Transpose::Trans);
            const OperandForm none = as_given(Core::Transpose::NoTrans);

            expect_calls_give_the_same_bits([&] { return run_gemm(conj, none, {}); },
                                            [&] { return run_gemm(trans, none, {}); });
            expect_calls_give_the_same_bits([&] { return run_gemm(none, conj, {}); },
                                            [&] { return run_gemm(none, trans, {}); });
            expect_calls_give_the_same_bits([&] { return run_gemm(conj, conj, {K_CASE_BATCH}); },
                                            [&] { return run_gemm(trans, trans, {K_CASE_BATCH}); });
        }
    } // namespace Test
} // namespace SushiBLAS
