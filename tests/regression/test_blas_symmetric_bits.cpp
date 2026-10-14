/**************************************************************************/
/* test_blas_symmetric_bits.cpp                                           */
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
 * @file test_blas_symmetric_bits.cpp
 * @brief Tests that the unconjugated reference kernels equal a plain host loop bit for bit.
 * @author Mustafa Garip
 */

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include "../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Seed of the data every case draws.
        constexpr std::uint32_t K_BITS_SEED = 20261007u;

        /// @brief Elements of the vectors of the dot product.
        constexpr int64_t K_BITS_VECTOR_LENGTH = 257;

        /// @brief Rows of the rank-1 update and of the right-hand sides of the solve.
        constexpr int64_t K_BITS_ROWS = 13;

        /// @brief Columns of the rank-1 update and of the right-hand sides of the solve.
        constexpr int64_t K_BITS_COLS = 9;

        /// @brief Order of the symmetric matrices.
        constexpr int64_t K_BITS_ORDER = 37;

        /// @brief Order of the result of the rank-k update.
        constexpr int64_t K_BITS_RANK_K_ORDER = 11;

        /// @brief Terms each element of the rank-k update sums.
        constexpr int64_t K_BITS_RANK_K_TERMS = 23;

        /// @brief Added to the diagonal of a triangular matrix, so a solve stays well scaled.
        constexpr double K_BITS_DIAGONAL_SHIFT = 4.0;

        /// @brief Both element orders a tensor can have.
        constexpr std::array<Core::Layout, 2> K_BITS_LAYOUTS = {Core::Layout::RowMajor,
                                                                Core::Layout::ColumnMajor};

        /// @brief Both triangles a symmetric or triangular call can name.
        constexpr std::array<Core::Uplo, 2> K_BITS_TRIANGLES = {Core::Uplo::Upper,
                                                                Core::Uplo::Lower};

        /// @brief The transposes that do not conjugate.
        constexpr std::array<Core::Transpose, 2> K_BITS_TRANSPOSES = {Core::Transpose::NoTrans,
                                                                      Core::Transpose::Trans};

        /**
         * @brief Runs a BLAS call on the reference kernel and computes the same result on the host.
         */
        template<typename T>
        class Regression_BlasSymmetricBits : public SushiBLASTest
        {
            protected:
                /** @brief The type a rank update is formed in. */
                using W = Kernels::Accumulator<T>;

                /** @brief The type a term of a sum is formed in. */
                using RW = Kernels::ReferenceAccumulator<T>;

                /** @brief Rebuilds the engine in layout on the reference path and reseeds. */
                void start(Core::Layout layout)
                {
                    reinit_engine(layout);
                    engine->set_mode(ExecutionMode::ForceReference);
                    generator_.seed(K_BITS_SEED);
                }

                /** @brief Returns real + imaginary * i in T; a real T drops the imaginary part. */
                [[nodiscard]] static T make(double real, [[maybe_unused]] double imaginary) noexcept
                {
                    if constexpr (Core::is_complex_v<T>)
                        return T(real, imaginary);
                    else
                        return static_cast<T>(real);
                }

                /** @brief Returns the scalar that multiplies the product or the update. */
                [[nodiscard]] static T alpha() noexcept
                {
                    return make(1.75, -0.625);
                }

                /** @brief Returns the scalar that multiplies the prior result. */
                [[nodiscard]] static T beta() noexcept
                {
                    return make(-0.625, 0.5);
                }

                /** @brief Returns count values with every part drawn from [-1, 1]. */
                [[nodiscard]] std::vector<T> random_values(int64_t count)
                {
                    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                    std::vector<T> values(index(count));
                    for (T& value : values)
                    {
                        const double real = distribution(generator_);
                        const double imaginary = distribution(generator_);
                        value = make(real, imaginary);
                    }
                    return values;
                }

                /** @brief Returns an n x n matrix of random_values with a shifted diagonal. */
                [[nodiscard]] std::vector<T> random_triangular(int64_t n)
                {
                    std::vector<T> values = random_values(n * n);
                    for (int64_t i = 0; i < n; ++i)
                        values[offset(i, i, n, n)] += make(K_BITS_DIAGONAL_SHIFT, 0.0);
                    return values;
                }

                /** @brief Returns a tensor of dims holding values in storage order. */
                [[nodiscard]] Tensor tensor_of(std::initializer_list<int64_t> dims,
                                               const std::vector<T>& values)
                {
                    return create_initialized_tensor<T>(dims, values);
                }

                /** @brief Returns i as a container index. */
                [[nodiscard]] static std::size_t index(int64_t i) noexcept
                {
                    return static_cast<std::size_t>(i);
                }

                /** @brief Returns where element (r, c) of a rows x cols matrix is stored. */
                [[nodiscard]] std::size_t offset(int64_t r, int64_t c, int64_t rows,
                                                 int64_t cols) const
                {
                    return (engine->layout() == Core::Layout::RowMajor) ? index(r * cols + c)
                                                                             : index(r + c * rows);
                }

                /** @brief Returns sum(x_i * y_i), summed in a ReferenceSum and rounded once. */
                [[nodiscard]] static T host_dot(const std::vector<T>& x, const std::vector<T>& y)
                {
                    Kernels::ReferenceSum<T> sum;
                    for (std::size_t i = 0; i < x.size(); ++i)
                        sum.add(static_cast<RW>(x[i]) * static_cast<RW>(y[i]));
                    return static_cast<T>(sum.value());
                }

                /** @brief Adds scale * x * y^T to the m x n matrix a. */
                void host_ger(int64_t m, int64_t n, T scale, const std::vector<T>& x,
                              const std::vector<T>& y, std::vector<T>& a) const
                {
                    for (int64_t i = 0; i < m; ++i)
                    {
                        for (int64_t j = 0; j < n; ++j)
                        {
                            T& element = a[offset(i, j, m, n)];
                            element = T(W(element) + W(scale) * W(x[index(i)]) * W(y[index(j)]));
                        }
                    }
                }

                /** @brief Overwrites y with scale * A * x + prior * y from one triangle of a. */
                void host_symv(Core::Uplo uplo, int64_t n, T scale, const std::vector<T>& a,
                               const std::vector<T>& x, T prior, std::vector<T>& y) const
                {
                    const bool upper = (uplo == Core::Uplo::Upper);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        Kernels::ReferenceSum<T> sum;
                        for (int64_t j = 0; j < n; ++j)
                        {
                            const bool stored = upper ? (i <= j) : (i >= j);
                            const T element = stored ? a[offset(i, j, n, n)]
                                                     : a[offset(j, i, n, n)];
                            sum.add(RW(element) * RW(x[index(j)]));
                        }

                        T& out = y[index(i)];
                        out = (prior == T(0)) ? T(RW(scale) * sum.value())
                                              : T(RW(scale) * sum.value() + RW(prior) * RW(out));
                    }
                }

                /** @brief Adds scale * x * x^T to one triangle of the n x n matrix a. */
                void host_syr(Core::Uplo uplo, int64_t n, T scale, const std::vector<T>& x,
                              std::vector<T>& a) const
                {
                    const bool upper = (uplo == Core::Uplo::Upper);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        const int64_t first = upper ? i : 0;
                        const int64_t last = upper ? n : i + 1;
                        for (int64_t j = first; j < last; ++j)
                        {
                            T& element = a[offset(i, j, n, n)];
                            element = T(W(element) + W(scale) * W(x[index(i)]) * W(x[index(j)]));
                        }
                    }
                }

                /** @brief Adds scale * (x * y^T + y * x^T) to one triangle of the matrix a. */
                void host_syr2(Core::Uplo uplo, int64_t n, T scale, const std::vector<T>& x,
                               const std::vector<T>& y, std::vector<T>& a) const
                {
                    const bool upper = (uplo == Core::Uplo::Upper);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        const int64_t first = upper ? i : 0;
                        const int64_t last = upper ? n : i + 1;
                        for (int64_t j = first; j < last; ++j)
                        {
                            T& element = a[offset(i, j, n, n)];
                            element = T(W(element) +
                                        W(scale) * (W(x[index(i)]) * W(y[index(j)]) +
                                                    W(y[index(i)]) * W(x[index(j)])));
                        }
                    }
                }

                /**
                 * @brief Overwrites one triangle of the n x n matrix c with
                 *        scale * op(A) * op(A)^T + prior * c, each element in a ReferenceSum.
                 * @param a n x k when trans is NO_TRANS, k x n when it is TRANS.
                 */
                void host_syrk(Core::Uplo uplo, Core::Transpose trans, int64_t n, int64_t k,
                               T scale, const std::vector<T>& a, T prior, std::vector<T>& c) const
                {
                    const bool upper = (uplo == Core::Uplo::Upper);
                    const bool transposed = (trans == Core::Transpose::Trans);
                    auto op_a = [&](int64_t r, int64_t l) -> T
                    {
                        return transposed ? a[offset(l, r, k, n)] : a[offset(r, l, n, k)];
                    };

                    for (int64_t i = 0; i < n; ++i)
                    {
                        for (int64_t j = 0; j < n; ++j)
                        {
                            if ((upper && j < i) || (!upper && j > i))
                                continue;

                            Kernels::ReferenceSum<T> sum;
                            for (int64_t l = 0; l < k; ++l)
                                sum.add(RW(op_a(i, l)) * RW(op_a(j, l)));

                            T& out = c[offset(i, j, n, n)];
                            out = (prior == T(0))
                                ? T(RW(scale) * sum.value())
                                : T(RW(scale) * sum.value() + RW(prior) * RW(out));
                        }
                    }
                }

                /**
                 * @brief Overwrites the m x n matrix b with the solution of op(A) * X = scale * B
                 *        or X * op(A) = scale * B, each step summed in a ReferenceSum.
                 * @param a Square, of order m on the left side and n on the right.
                 */
                void host_trsm(Core::Side side, Core::Uplo uplo, Core::Transpose trans,
                               Core::Diag diag, int64_t m, int64_t n, T scale,
                               const std::vector<T>& a, std::vector<T>& b) const
                {
                    const bool left = (side == Core::Side::Left);
                    const bool upper = (uplo == Core::Uplo::Upper);
                    const bool transposed = (trans == Core::Transpose::Trans);
                    const bool unit_diagonal = (diag == Core::Diag::Unit);
                    const int64_t order = left ? m : n;

                    // On the right side a row of X solves against the transpose of op(A).
                    const bool swapped = left ? transposed : !transposed;
                    const bool lower = swapped ? upper : !upper;
                    auto factor = [&](int64_t r, int64_t c) -> T
                    {
                        return swapped ? a[offset(c, r, order, order)]
                                       : a[offset(r, c, order, order)];
                    };
                    auto unknown = [&](int64_t step, int64_t line) -> T&
                    {
                        return left ? b[offset(step, line, m, n)] : b[offset(line, step, m, n)];
                    };

                    const int64_t lines = left ? n : m;
                    for (int64_t line = 0; line < lines; ++line)
                    {
                        for (int64_t count = 0; count < order; ++count)
                        {
                            const int64_t step = lower ? count : order - 1 - count;
                            const int64_t first = lower ? 0 : step + 1;
                            const int64_t last = lower ? step : order;

                            Kernels::ReferenceSum<T> sum;
                            sum.add(RW(scale) * RW(unknown(step, line)));
                            for (int64_t solved = first; solved < last; ++solved)
                                sum.add(-(RW(factor(step, solved)) * RW(unknown(solved, line))));

                            unknown(step, line) = unit_diagonal
                                ? T(sum.value())
                                : T(sum.value() / RW(factor(step, step)));
                        }
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

            private:
                /// @brief The generator every random_values call advances.
                std::mt19937 generator_{K_BITS_SEED};
        };

        using SymmetricBitsTypes = ::testing::Types<float, double, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Regression_BlasSymmetricBits, SymmetricBitsTypes);

        TYPED_TEST(Regression_BlasSymmetricBits, DotEqualsTheHostSum)
        {
            using T = TypeParam;

            for (const Core::Layout layout : K_BITS_LAYOUTS)
            {
                this->start(layout);
                const std::vector<T> x = this->random_values(K_BITS_VECTOR_LENGTH);
                const std::vector<T> y = this->random_values(K_BITS_VECTOR_LENGTH);
                Tensor x_tensor = this->tensor_of({K_BITS_VECTOR_LENGTH}, x);
                Tensor y_tensor = this->tensor_of({K_BITS_VECTOR_LENGTH}, y);
                Tensor result = this->engine->create_tensor({1}, sushiblas_dtype<T>());

                SushiBLAS::BLASOps ops(*this->engine);
                if constexpr (Core::is_complex_v<T>)
                    ops.dotu(x_tensor, y_tensor, result);
                else
                    ops.dot(x_tensor, y_tensor, result);
                this->engine->execute().value();

                const std::vector<T> expected = {TestFixture::host_dot(x, y)};
                TestFixture::expect_same_bits(read_tensor<T>(result), expected);
            }
        }

        TYPED_TEST(Regression_BlasSymmetricBits, GerEqualsTheHostUpdate)
        {
            using T = TypeParam;
            const T alpha = TestFixture::alpha();

            for (const Core::Layout layout : K_BITS_LAYOUTS)
            {
                this->start(layout);
                const std::vector<T> x = this->random_values(K_BITS_ROWS);
                const std::vector<T> y = this->random_values(K_BITS_COLS);
                std::vector<T> a = this->random_values(K_BITS_ROWS * K_BITS_COLS);
                Tensor x_tensor = this->tensor_of({K_BITS_ROWS}, x);
                Tensor y_tensor = this->tensor_of({K_BITS_COLS}, y);
                Tensor a_tensor = this->tensor_of({K_BITS_ROWS, K_BITS_COLS}, a);

                SushiBLAS::BLASOps ops(*this->engine);
                if constexpr (Core::is_complex_v<T>)
                    ops.geru(x_tensor, y_tensor, a_tensor, BLASScalar(alpha));
                else
                    ops.ger(x_tensor, y_tensor, a_tensor, static_cast<double>(alpha));
                this->engine->execute().value();

                this->host_ger(K_BITS_ROWS, K_BITS_COLS, alpha, x, y, a);
                TestFixture::expect_same_bits(read_tensor<T>(a_tensor), a);
            }
        }

        TYPED_TEST(Regression_BlasSymmetricBits, SymvEqualsTheHostSum)
        {
            using T = TypeParam;
            const T alpha = TestFixture::alpha();

            for (const Core::Layout layout : K_BITS_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                {
                    for (const T beta : {TestFixture::beta(), T(0)})
                    {
                        this->start(layout);
                        const std::vector<T> a = this->random_values(K_BITS_ORDER * K_BITS_ORDER);
                        const std::vector<T> x = this->random_values(K_BITS_ORDER);
                        std::vector<T> y = this->random_values(K_BITS_ORDER);
                        Tensor a_tensor = this->tensor_of({K_BITS_ORDER, K_BITS_ORDER}, a);
                        Tensor x_tensor = this->tensor_of({K_BITS_ORDER}, x);
                        Tensor y_tensor = this->tensor_of({K_BITS_ORDER}, y);

                        SushiBLAS::BLASOps(*this->engine).symv(a_tensor, x_tensor, y_tensor, uplo,
                                                               BLASScalar(alpha), BLASScalar(beta));
                        this->engine->execute().value();

                        this->host_symv(uplo, K_BITS_ORDER, alpha, a, x, beta, y);
                        TestFixture::expect_same_bits(read_tensor<T>(y_tensor), y);
                    }
                }
            }
        }

        TYPED_TEST(Regression_BlasSymmetricBits, SyrEqualsTheHostUpdate)
        {
            using T = TypeParam;
            const T alpha = TestFixture::alpha();

            for (const Core::Layout layout : K_BITS_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                {
                    this->start(layout);
                    const std::vector<T> x = this->random_values(K_BITS_ORDER);
                    std::vector<T> a = this->random_values(K_BITS_ORDER * K_BITS_ORDER);
                    Tensor x_tensor = this->tensor_of({K_BITS_ORDER}, x);
                    Tensor a_tensor = this->tensor_of({K_BITS_ORDER, K_BITS_ORDER}, a);

                    SushiBLAS::BLASOps(*this->engine).syr(x_tensor, a_tensor, uplo,
                                                          BLASScalar(alpha));
                    this->engine->execute().value();

                    this->host_syr(uplo, K_BITS_ORDER, alpha, x, a);
                    TestFixture::expect_same_bits(read_tensor<T>(a_tensor), a);
                }
            }
        }

        TYPED_TEST(Regression_BlasSymmetricBits, Syr2EqualsTheHostUpdate)
        {
            using T = TypeParam;
            const T alpha = TestFixture::alpha();

            for (const Core::Layout layout : K_BITS_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                {
                    this->start(layout);
                    const std::vector<T> x = this->random_values(K_BITS_ORDER);
                    const std::vector<T> y = this->random_values(K_BITS_ORDER);
                    std::vector<T> a = this->random_values(K_BITS_ORDER * K_BITS_ORDER);
                    Tensor x_tensor = this->tensor_of({K_BITS_ORDER}, x);
                    Tensor y_tensor = this->tensor_of({K_BITS_ORDER}, y);
                    Tensor a_tensor = this->tensor_of({K_BITS_ORDER, K_BITS_ORDER}, a);

                    SushiBLAS::BLASOps(*this->engine).syr2(x_tensor, y_tensor, a_tensor, uplo,
                                                           BLASScalar(alpha));
                    this->engine->execute().value();

                    this->host_syr2(uplo, K_BITS_ORDER, alpha, x, y, a);
                    TestFixture::expect_same_bits(read_tensor<T>(a_tensor), a);
                }
            }
        }

        TYPED_TEST(Regression_BlasSymmetricBits, SyrkEqualsTheHostSum)
        {
            using T = TypeParam;
            const T alpha = TestFixture::alpha();
            const int64_t n = K_BITS_RANK_K_ORDER;
            const int64_t k = K_BITS_RANK_K_TERMS;

            for (const Core::Layout layout : K_BITS_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                {
                    for (const Core::Transpose trans : K_BITS_TRANSPOSES)
                    {
                        for (const T beta : {TestFixture::beta(), T(0)})
                        {
                            const bool transposed = (trans == Core::Transpose::Trans);

                            this->start(layout);
                            const std::vector<T> a = this->random_values(n * k);
                            std::vector<T> c = this->random_values(n * n);
                            Tensor a_tensor = transposed ? this->tensor_of({k, n}, a)
                                                         : this->tensor_of({n, k}, a);
                            Tensor c_tensor = this->tensor_of({n, n}, c);

                            SushiBLAS::BLASOps(*this->engine).syrk(a_tensor, c_tensor, uplo, trans,
                                                                   BLASScalar(alpha),
                                                                   BLASScalar(beta));
                            this->engine->execute().value();

                            this->host_syrk(uplo, trans, n, k, alpha, a, beta, c);
                            TestFixture::expect_same_bits(read_tensor<T>(c_tensor), c);
                        }
                    }
                }
            }
        }

        TYPED_TEST(Regression_BlasSymmetricBits, TrsmEqualsTheHostSubstitution)
        {
            using T = TypeParam;
            const T alpha = TestFixture::alpha();
            const int64_t m = K_BITS_ROWS;
            const int64_t n = K_BITS_COLS;

            for (const Core::Layout layout : K_BITS_LAYOUTS)
            {
                for (const Core::Side side : {Core::Side::Left, Core::Side::Right})
                {
                    for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                    {
                        for (const Core::Transpose trans : K_BITS_TRANSPOSES)
                        {
                            for (const Core::Diag diag : {Core::Diag::NonUnit, Core::Diag::Unit})
                            {
                                const int64_t order = (side == Core::Side::Left) ? m : n;

                                this->start(layout);
                                const std::vector<T> a = this->random_triangular(order);
                                std::vector<T> b = this->random_values(m * n);
                                Tensor a_tensor = this->tensor_of({order, order}, a);
                                Tensor b_tensor = this->tensor_of({m, n}, b);

                                SushiBLAS::BLASOps(*this->engine).trsm(a_tensor, b_tensor, side,
                                                                       uplo, trans, diag,
                                                                       BLASScalar(alpha));
                                this->engine->execute().value();

                                this->host_trsm(side, uplo, trans, diag, m, n, alpha, a, b);
                                TestFixture::expect_same_bits(read_tensor<T>(b_tensor), b);
                            }
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
