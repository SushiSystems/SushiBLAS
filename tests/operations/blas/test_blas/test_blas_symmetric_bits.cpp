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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
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

            /// @brief The element types the unconjugated kernels are checked in.
            const std::vector<Core::DataType> K_BITS_TYPES = {
                Core::DataType::Float32, Core::DataType::Float64, Core::DataType::Complex64};

            /// @brief The one kernel the bits are compared on.
            const std::vector<ExecutionMode> K_BITS_MODE = {ExecutionMode::ForceReference};

            /// @brief Both triangles a symmetric or triangular call can name.
            const std::vector<Core::Uplo> K_BITS_TRIANGLES = {Core::Uplo::Upper,
                                                              Core::Uplo::Lower};

            /// @brief The transposes that do not conjugate.
            constexpr std::array<Core::Transpose, 2> K_BITS_TRANSPOSES = {Core::Transpose::NoTrans,
                                                                          Core::Transpose::Trans};

            /** @brief Names a triangle and whether the prior result is scaled by zero. */
            struct UploBeta
            {
                Core::Uplo uplo;  /**< The triangle the call names. */
                bool zero_beta;   /**< Whether beta is zero, so the prior result is dropped. */
            };

            /** @brief Names the triangle, the form of A and the scale of the prior C. */
            struct RankKForm
            {
                Core::Uplo uplo;         /**< The triangle the call updates. */
                Core::Transpose trans;   /**< The form of A. */
                bool zero_beta;          /**< Whether beta is zero, so the prior C is dropped. */
            };

            /** @brief Names the side, triangle, form of A and diagonal of a solve. */
            struct SolveForm
            {
                Core::Side side;         /**< The side A stands on. */
                Core::Uplo uplo;         /**< The triangle of A. */
                Core::Transpose trans;   /**< The form of A. */
                Core::Diag diag;         /**< How the diagonal of A is read. */
            };

            /** @brief Returns the tail of a row named by a triangle and a beta. */
            [[nodiscard]] std::string axis_name(const UploBeta& form)
            {
                return "_" + word(form.uplo) + (form.zero_beta ? "_BetaZero" : "_BetaSet");
            }

            /** @brief Returns the tail of a row named by a rank-k update. */
            [[nodiscard]] std::string axis_name(const RankKForm& form)
            {
                return "_" + word(form.uplo) + "_" + word(form.trans) +
                       (form.zero_beta ? "_BetaZero" : "_BetaSet");
            }

            /** @brief Returns the tail of a row named by a solve. */
            [[nodiscard]] std::string axis_name(const SolveForm& form)
            {
                return "_" + word(form.side) + "_" + word(form.uplo) + "_" + word(form.trans) +
                       "_" + word(form.diag);
            }

            /** @brief Returns every triangle with a set and a zero beta. */
            [[nodiscard]] std::vector<UploBeta> symv_forms()
            {
                std::vector<UploBeta> forms;
                for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                    for (const bool zero_beta : {false, true})
                        forms.push_back({uplo, zero_beta});
                return forms;
            }

            /** @brief Returns every triangle, form of A and beta of a rank-k update. */
            [[nodiscard]] std::vector<RankKForm> syrk_forms()
            {
                std::vector<RankKForm> forms;
                for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                    for (const Core::Transpose trans : K_BITS_TRANSPOSES)
                        for (const bool zero_beta : {false, true})
                            forms.push_back({uplo, trans, zero_beta});
                return forms;
            }

            /** @brief Returns every side, triangle, form of A and diagonal of a solve. */
            [[nodiscard]] std::vector<SolveForm> trsm_forms()
            {
                std::vector<SolveForm> forms;
                for (const Core::Side side : {Core::Side::Left, Core::Side::Right})
                    for (const Core::Uplo uplo : K_BITS_TRIANGLES)
                        for (const Core::Transpose trans : K_BITS_TRANSPOSES)
                            for (const Core::Diag diag : {Core::Diag::NonUnit, Core::Diag::Unit})
                                forms.push_back({side, uplo, trans, diag});
                return forms;
            }

            /**
             * @brief Draws the data of a case and computes the result of each unconjugated call
             *        on the host, in the order the reference kernels sum.
             * @tparam T The element type of the call.
             */
            template<typename T>
            class SymmetricHostOracle
            {
                public:
                    /** @brief The type a rank update is formed in. */
                    using W = Kernels::Accumulator<T>;

                    /** @brief The type a term of a sum is formed in. */
                    using RW = Kernels::ReferenceAccumulator<T>;

                    /** @brief Builds an oracle for matrices stored in layout, seeded afresh. */
                    explicit SymmetricHostOracle(Core::Layout layout)
                        : layout_(layout), generator_(K_BITS_SEED)
                    {
                    }

                    /** @brief Returns real + imaginary * i in T; a real T drops the imaginary part. */
                    [[nodiscard]] static T make(double real,
                                                [[maybe_unused]] double imaginary) noexcept
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

                    /** @brief Returns the scalar that multiplies the prior result, or zero. */
                    [[nodiscard]] static T beta(bool zero) noexcept
                    {
                        return zero ? T(0) : make(-0.625, 0.5);
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
                                   T scale, const std::vector<T>& a, T prior,
                                   std::vector<T>& c) const
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
                    /** @brief Returns i as a container index. */
                    [[nodiscard]] static std::size_t index(int64_t i) noexcept
                    {
                        return static_cast<std::size_t>(i);
                    }

                    /** @brief Returns where element (r, c) of a rows x cols matrix is stored. */
                    [[nodiscard]] std::size_t offset(int64_t r, int64_t c, int64_t rows,
                                                     int64_t cols) const
                    {
                        return (layout_ == Core::Layout::RowMajor) ? index(r * cols + c)
                                                                   : index(r + c * rows);
                    }

                    /// @brief The element order of every matrix.
                    Core::Layout layout_;

                    /// @brief The generator every random_values call advances.
                    std::mt19937 generator_;
            };
        } // namespace

        /** @brief Builds the tensors of a symmetric bit comparison. */
        class BlasSymmetricBitsFixture : public BlasEngineStart
        {
            protected:
                /** @brief Returns a tensor of dims holding values in storage order. */
                template<typename T>
                [[nodiscard]] Tensor tensor_of(std::initializer_list<int64_t> dims,
                                               const std::vector<T>& values)
                {
                    return create_initialized_tensor<T>(dims, values);
                }
        };

        /**
         * @brief Runs a BLAS call on the reference kernel of the row, in the layout and element
         *        type of the row, and compares it with the same result computed on the host.
         * @tparam Axis NoAxis, or the type of the value that completes each row.
         */
        template<typename Axis>
        using BlasSymmetricBitsCase = BlasRowCase<BlasSymmetricBitsFixture, Axis>;

        /** @brief Runs the cases that take no axis beyond layout and element type. */
        class Medium_BlasSymmetricBits : public BlasSymmetricBitsCase<NoAxis>
        {

        };

        /** @brief Runs the rank-1 updates of one triangle. */
        class Medium_BlasSymmetricBitsTriangle : public BlasSymmetricBitsCase<Core::Uplo>
        {

        };

        /** @brief Runs the symmetric product under each triangle and beta. */
        class Medium_BlasSymmetricBitsSymv : public BlasSymmetricBitsCase<UploBeta>
        {

        };

        /** @brief Runs the rank-k update under each triangle, form of A and beta. */
        class Medium_BlasSymmetricBitsSyrk : public BlasSymmetricBitsCase<RankKForm>
        {

        };

        /** @brief Runs the triangular solve under each side, triangle, form and diagonal. */
        class Medium_BlasSymmetricBitsTrsm : public BlasSymmetricBitsCase<SolveForm>
        {

        };

        /** @brief Tests that dot, or dotu on a complex type, equals the host sum bit for bit. */
        TEST_P(Medium_BlasSymmetricBits, DotEqualsTheHostSum)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Host = SymmetricHostOracle<T>;
                Host host(GetParam().layout);

                const std::vector<T> x = host.random_values(K_BITS_VECTOR_LENGTH);
                const std::vector<T> y = host.random_values(K_BITS_VECTOR_LENGTH);
                Tensor x_tensor = tensor_of({K_BITS_VECTOR_LENGTH}, x);
                Tensor y_tensor = tensor_of({K_BITS_VECTOR_LENGTH}, y);
                Tensor result = engine->create_tensor({1}, sushiblas_dtype<T>());

                SushiBLAS::BLASOps ops(recorder());
                if constexpr (Core::is_complex_v<T>)
                    ops.dotu(x_tensor, y_tensor, result);
                else
                    ops.dot(x_tensor, y_tensor, result);
                recording->run().value();

                const std::vector<T> expected = {Host::host_dot(x, y)};
                Host::expect_same_bits(read_tensor<T>(result), expected);
            });
        }

        /** @brief Tests that ger, or geru on a complex type, equals the host update bit for bit. */
        TEST_P(Medium_BlasSymmetricBits, GerEqualsTheHostUpdate)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Host = SymmetricHostOracle<T>;
                Host host(GetParam().layout);
                const T alpha = Host::alpha();

                const std::vector<T> x = host.random_values(K_BITS_ROWS);
                const std::vector<T> y = host.random_values(K_BITS_COLS);
                std::vector<T> a = host.random_values(K_BITS_ROWS * K_BITS_COLS);
                Tensor x_tensor = tensor_of({K_BITS_ROWS}, x);
                Tensor y_tensor = tensor_of({K_BITS_COLS}, y);
                Tensor a_tensor = tensor_of({K_BITS_ROWS, K_BITS_COLS}, a);

                SushiBLAS::BLASOps ops(recorder());
                if constexpr (Core::is_complex_v<T>)
                    ops.geru(x_tensor, y_tensor, a_tensor, BLASScalar(alpha));
                else
                    ops.ger(x_tensor, y_tensor, a_tensor, static_cast<double>(alpha));
                recording->run().value();

                host.host_ger(K_BITS_ROWS, K_BITS_COLS, alpha, x, y, a);
                Host::expect_same_bits(read_tensor<T>(a_tensor), a);
            });
        }

        /** @brief Tests that symv equals the host sum bit for bit. */
        TEST_P(Medium_BlasSymmetricBitsSymv, SymvEqualsTheHostSum)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Host = SymmetricHostOracle<T>;
                Host host(GetParam().layout);
                const Core::Uplo uplo = GetParam().axis.uplo;
                const T alpha = Host::alpha();
                const T beta = Host::beta(GetParam().axis.zero_beta);

                const std::vector<T> a = host.random_values(K_BITS_ORDER * K_BITS_ORDER);
                const std::vector<T> x = host.random_values(K_BITS_ORDER);
                std::vector<T> y = host.random_values(K_BITS_ORDER);
                Tensor a_tensor = tensor_of({K_BITS_ORDER, K_BITS_ORDER}, a);
                Tensor x_tensor = tensor_of({K_BITS_ORDER}, x);
                Tensor y_tensor = tensor_of({K_BITS_ORDER}, y);

                SushiBLAS::BLASOps(recorder())
                    .symv(a_tensor, x_tensor, y_tensor, uplo, BLASScalar(alpha), BLASScalar(beta));
                recording->run().value();

                host.host_symv(uplo, K_BITS_ORDER, alpha, a, x, beta, y);
                Host::expect_same_bits(read_tensor<T>(y_tensor), y);
            });
        }

        /** @brief Tests that syr equals the host update bit for bit. */
        TEST_P(Medium_BlasSymmetricBitsTriangle, SyrEqualsTheHostUpdate)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Host = SymmetricHostOracle<T>;
                Host host(GetParam().layout);
                const Core::Uplo uplo = GetParam().axis;
                const T alpha = Host::alpha();

                const std::vector<T> x = host.random_values(K_BITS_ORDER);
                std::vector<T> a = host.random_values(K_BITS_ORDER * K_BITS_ORDER);
                Tensor x_tensor = tensor_of({K_BITS_ORDER}, x);
                Tensor a_tensor = tensor_of({K_BITS_ORDER, K_BITS_ORDER}, a);

                SushiBLAS::BLASOps(recorder()).syr(x_tensor, a_tensor, uplo, BLASScalar(alpha));
                recording->run().value();

                host.host_syr(uplo, K_BITS_ORDER, alpha, x, a);
                Host::expect_same_bits(read_tensor<T>(a_tensor), a);
            });
        }

        /** @brief Tests that syr2 equals the host update bit for bit. */
        TEST_P(Medium_BlasSymmetricBitsTriangle, Syr2EqualsTheHostUpdate)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Host = SymmetricHostOracle<T>;
                Host host(GetParam().layout);
                const Core::Uplo uplo = GetParam().axis;
                const T alpha = Host::alpha();

                const std::vector<T> x = host.random_values(K_BITS_ORDER);
                const std::vector<T> y = host.random_values(K_BITS_ORDER);
                std::vector<T> a = host.random_values(K_BITS_ORDER * K_BITS_ORDER);
                Tensor x_tensor = tensor_of({K_BITS_ORDER}, x);
                Tensor y_tensor = tensor_of({K_BITS_ORDER}, y);
                Tensor a_tensor = tensor_of({K_BITS_ORDER, K_BITS_ORDER}, a);

                SushiBLAS::BLASOps(recorder())
                    .syr2(x_tensor, y_tensor, a_tensor, uplo, BLASScalar(alpha));
                recording->run().value();

                host.host_syr2(uplo, K_BITS_ORDER, alpha, x, y, a);
                Host::expect_same_bits(read_tensor<T>(a_tensor), a);
            });
        }

        /** @brief Tests that syrk equals the host sum bit for bit. */
        TEST_P(Medium_BlasSymmetricBitsSyrk, SyrkEqualsTheHostSum)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Host = SymmetricHostOracle<T>;
                Host host(GetParam().layout);
                const Core::Uplo uplo = GetParam().axis.uplo;
                const Core::Transpose trans = GetParam().axis.trans;
                const T alpha = Host::alpha();
                const T beta = Host::beta(GetParam().axis.zero_beta);
                const int64_t n = K_BITS_RANK_K_ORDER;
                const int64_t k = K_BITS_RANK_K_TERMS;
                const bool transposed = (trans == Core::Transpose::Trans);

                const std::vector<T> a = host.random_values(n * k);
                std::vector<T> c = host.random_values(n * n);
                Tensor a_tensor = transposed ? tensor_of({k, n}, a) : tensor_of({n, k}, a);
                Tensor c_tensor = tensor_of({n, n}, c);

                SushiBLAS::BLASOps(recorder())
                    .syrk(a_tensor, c_tensor, uplo, trans, BLASScalar(alpha), BLASScalar(beta));
                recording->run().value();

                host.host_syrk(uplo, trans, n, k, alpha, a, beta, c);
                Host::expect_same_bits(read_tensor<T>(c_tensor), c);
            });
        }

        /** @brief Tests that trsm equals the host substitution bit for bit. */
        TEST_P(Medium_BlasSymmetricBitsTrsm, TrsmEqualsTheHostSubstitution)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Host = SymmetricHostOracle<T>;
                Host host(GetParam().layout);
                const SolveForm form = GetParam().axis;
                const T alpha = Host::alpha();
                const int64_t m = K_BITS_ROWS;
                const int64_t n = K_BITS_COLS;
                const int64_t order = (form.side == Core::Side::Left) ? m : n;

                const std::vector<T> a = host.random_triangular(order);
                std::vector<T> b = host.random_values(m * n);
                Tensor a_tensor = tensor_of({order, order}, a);
                Tensor b_tensor = tensor_of({m, n}, b);

                SushiBLAS::BLASOps(recorder())
                    .trsm(a_tensor, b_tensor, form.side, form.uplo, form.trans, form.diag,
                          BLASScalar(alpha));
                recording->run().value();

                host.host_trsm(form.side, form.uplo, form.trans, form.diag, m, n, alpha, a, b);
                Host::expect_same_bits(read_tensor<T>(b_tensor), b);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasSymmetricBits, ::testing::ValuesIn(plain_rows(K_BITS_TYPES, K_BITS_MODE)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasSymmetricBitsTriangle,
            ::testing::ValuesIn(rows_of(K_BITS_TYPES, K_BITS_MODE, K_BITS_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasSymmetricBitsSymv,
            ::testing::ValuesIn(rows_of(K_BITS_TYPES, K_BITS_MODE, symv_forms())),
            row_name<UploBeta>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasSymmetricBitsSyrk,
            ::testing::ValuesIn(rows_of(K_BITS_TYPES, K_BITS_MODE, syrk_forms())),
            row_name<RankKForm>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasSymmetricBitsTrsm,
            ::testing::ValuesIn(rows_of(K_BITS_TYPES, K_BITS_MODE, trsm_forms())),
            row_name<SolveForm>);
    } // namespace Test
} // namespace SushiBLAS
