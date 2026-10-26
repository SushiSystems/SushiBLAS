/**************************************************************************/
/* gmres.hpp                                                              */
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
 * @file gmres.hpp
 * @brief Defines the reference kernel of the restarted, block-Jacobi preconditioned GMRES solve.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/arithmetic/no_contraction.hpp>
#include <SushiBLAS/linalg_kernels/chunked_dot.hpp>
#include <SushiBLAS/linalg_kernels/gmres_workspace.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Detail
            {
                /**
                 * @brief Holds the host arrays of one reference GMRES solve and runs it.
                 * @see modules/kernels/linalg_kernels/README.md
                 */
                template<typename T>
                class GmresSolve final
                {
                    public:
                        /** @brief Sizes every array for @p layout. */
                        GmresSolve(const T* matrix, const GmresLayout& layout)
                            : matrix_(matrix), n_(layout.order), restart_(layout.restart),
                              block_(layout.block_size), values_(layout.scalars),
                              pivots_(layout.order), layout_(layout)
                        {
                        }

                        /**
                         * @brief Solves A x = b from the guess in @p x.
                         * @param iterations Element 0 receives the inner iterations taken.
                         * @param residual Element 0 receives the true |b - A x| / |b|.
                         */
                        void solve(const T* rhs, T* x, std::uint32_t* iterations, T* residual,
                                   const GmresSettings& settings)
                        {
                            factor_blocks();
                            iterations[0] = 0U;
                            residual[0] = T(0);
                            const T rhs_norm = norm(rhs);
                            if (rhs_norm == T(0))
                            {
                                for (std::size_t i = 0; i < n_; ++i)
                                    x[i] = T(0);
                                return;
                            }

                            const T tolerance = static_cast<T>(settings.relative_tolerance);
                            std::size_t taken = 0;
                            for (;;)
                            {
                                multiply(x, at(layout_.work));
                                for (std::size_t i = 0; i < n_; ++i)
                                    at(layout_.residual)[i] = rhs[i] - at(layout_.work)[i];
                                const T beta = norm(at(layout_.residual));
                                residual[0] = beta / rhs_norm;
                                if (residual[0] <= tolerance || taken >= settings.max_iterations)
                                    break;
                                taken += cycle(x, beta, rhs_norm, tolerance,
                                               settings.max_iterations - taken);
                            }
                            iterations[0] = static_cast<std::uint32_t>(taken);
                        }

                    private:
                        /** @brief Returns the array that starts at @p offset. */
                        T* at(std::size_t offset) noexcept
                        {
                            return values_.data() + offset;
                        }

                        /** @brief Returns the Hessenberg element at @p row, @p column. */
                        T& h(std::size_t row, std::size_t column) noexcept
                        {
                            return at(layout_.hessenberg)[row * restart_ + column];
                        }

                        /** @brief Returns basis vector @p index. */
                        T* vector(std::size_t index) noexcept
                        {
                            return at(layout_.basis) + index * n_;
                        }

                        /** @brief Returns the dot product of two n-vectors in chunked order. */
                        T dot(const T* a, const T* b) const noexcept
                        {
                            return chunked_dot(a, b, n_);
                        }

                        /** @brief Returns the Euclidean norm of an n-vector. */
                        T norm(const T* a) const noexcept
                        {
                            return std::sqrt(dot(a, a));
                        }

                        /** @brief Writes out = A in. */
                        void multiply(const T* in, T* out) const noexcept
                        {
                            for (std::size_t row = 0; row < n_; ++row)
                                out[row] = dot(matrix_ + row * n_, in);
                        }

                        /** @brief Factors every diagonal block of A by LU with partial pivoting. */
                        void factor_blocks()
                        {
                            SB_FP_CONTRACT_OFF
                            const std::size_t b = block_;
                            for (std::size_t start = 0; start < n_; start += b)
                            {
                                T* lu = at(layout_.factors) + start * b;
                                for (std::size_t r = 0; r < b; ++r)
                                    for (std::size_t c = 0; c < b; ++c)
                                        lu[r * b + c] = matrix_[(start + r) * n_ + start + c];
                                std::uint32_t* pivot = pivots_.data() + start;
                                for (std::size_t k = 0; k < b; ++k)
                                {
                                    std::size_t best = k;
                                    for (std::size_t r = k + 1; r < b; ++r)
                                        if (std::abs(lu[r * b + k]) > std::abs(lu[best * b + k]))
                                            best = r;
                                    pivot[k] = static_cast<std::uint32_t>(best);
                                    if (best != k)
                                        for (std::size_t c = 0; c < b; ++c)
                                        {
                                            const T held = lu[k * b + c];
                                            lu[k * b + c] = lu[best * b + c];
                                            lu[best * b + c] = held;
                                        }
                                    for (std::size_t r = k + 1; r < b; ++r)
                                    {
                                        const T factor = lu[r * b + k] / lu[k * b + k];
                                        lu[r * b + k] = factor;
                                        for (std::size_t c = k + 1; c < b; ++c)
                                        {
                                            const T product = factor * lu[k * b + c];
                                            lu[r * b + c] -= product;
                                        }
                                    }
                                }
                            }
                        }

                        /** @brief Writes out = M^-1 in, solving each block with its factors. */
                        void precondition(const T* in, T* out) noexcept
                        {
                            SB_FP_CONTRACT_OFF
                            const std::size_t b = block_;
                            for (std::size_t start = 0; start < n_; start += b)
                            {
                                const T* lu = at(layout_.factors) + start * b;
                                const std::uint32_t* pivot = pivots_.data() + start;
                                T* y = out + start;
                                for (std::size_t r = 0; r < b; ++r)
                                    y[r] = in[start + r];
                                for (std::size_t k = 0; k < b; ++k)
                                    if (pivot[k] != k)
                                    {
                                        const T held = y[k];
                                        y[k] = y[pivot[k]];
                                        y[pivot[k]] = held;
                                    }
                                for (std::size_t r = 1; r < b; ++r)
                                    for (std::size_t c = 0; c < r; ++c)
                                    {
                                        const T product = lu[r * b + c] * y[c];
                                        y[r] -= product;
                                    }
                                for (std::size_t r = b; r-- > 0;)
                                {
                                    for (std::size_t c = r + 1; c < b; ++c)
                                    {
                                        const T product = lu[r * b + c] * y[c];
                                        y[r] -= product;
                                    }
                                    y[r] /= lu[r * b + r];
                                }
                            }
                        }

                        /**
                         * @brief Runs one restart cycle from the residual of norm @p beta.
                         * @return The inner iterations taken, at most @p budget.
                         */
                        std::size_t cycle(T* x, T beta, T rhs_norm, T tolerance,
                                          std::size_t budget)
                        {
                            SB_FP_CONTRACT_OFF
                            T* work = at(layout_.work);
                            T* preconditioned = at(layout_.preconditioned);
                            T* cosines = at(layout_.cosines);
                            T* sines = at(layout_.sines);
                            T* projection = at(layout_.projection);
                            T* coefficients = at(layout_.coefficients);

                            for (std::size_t i = 0; i < n_; ++i)
                                vector(0)[i] = at(layout_.residual)[i] / beta;
                            for (std::size_t i = 0; i <= restart_; ++i)
                                projection[i] = T(0);
                            projection[0] = beta;

                            std::size_t steps = 0;
                            while (steps < restart_ && steps < budget)
                            {
                                const std::size_t j = steps;
                                precondition(vector(j), preconditioned);
                                multiply(preconditioned, work);
                                for (std::size_t i = 0; i <= j; ++i)
                                    h(i, j) = T(0);
                                // Modified Gram-Schmidt, run twice to restore lost orthogonality.
                                for (int pass = 0; pass < 2; ++pass)
                                    for (std::size_t i = 0; i <= j; ++i)
                                    {
                                        const T coefficient = dot(work, vector(i));
                                        h(i, j) += coefficient;
                                        for (std::size_t e = 0; e < n_; ++e)
                                        {
                                            const T product = coefficient * vector(i)[e];
                                            work[e] -= product;
                                        }
                                    }
                                const T next_norm = norm(work);
                                h(j + 1, j) = next_norm;
                                if (next_norm != T(0))
                                    for (std::size_t e = 0; e < n_; ++e)
                                        vector(j + 1)[e] = work[e] / next_norm;
                                for (std::size_t i = 0; i < j; ++i)
                                {
                                    const T cos_top = cosines[i] * h(i, j);
                                    const T sin_bottom = sines[i] * h(i + 1, j);
                                    const T sin_top = sines[i] * h(i, j);
                                    const T cos_bottom = cosines[i] * h(i + 1, j);
                                    h(i + 1, j) = cos_bottom - sin_top;
                                    h(i, j) = cos_top + sin_bottom;
                                }
                                const T top_square = h(j, j) * h(j, j);
                                const T bottom_square = h(j + 1, j) * h(j + 1, j);
                                const T radius = std::sqrt(top_square + bottom_square);
                                cosines[j] = radius == T(0) ? T(1) : h(j, j) / radius;
                                sines[j] = radius == T(0) ? T(0) : h(j + 1, j) / radius;
                                h(j, j) = radius;
                                h(j + 1, j) = T(0);
                                projection[j + 1] = -sines[j] * projection[j];
                                projection[j] = cosines[j] * projection[j];
                                ++steps;
                                if (std::abs(projection[j + 1]) / rhs_norm <= tolerance ||
                                    next_norm == T(0))
                                    break;
                            }

                            for (std::size_t r = steps; r-- > 0;)
                            {
                                T value = projection[r];
                                for (std::size_t c = r + 1; c < steps; ++c)
                                {
                                    const T product = h(r, c) * coefficients[c];
                                    value -= product;
                                }
                                coefficients[r] = h(r, r) == T(0) ? T(0) : value / h(r, r);
                            }
                            for (std::size_t e = 0; e < n_; ++e)
                            {
                                T value = T(0);
                                for (std::size_t k = 0; k < steps; ++k)
                                {
                                    const T product = coefficients[k] * vector(k)[e];
                                    value += product;
                                }
                                work[e] = value;
                            }
                            precondition(work, preconditioned);
                            for (std::size_t e = 0; e < n_; ++e)
                                x[e] += preconditioned[e];
                            return steps;
                        }

                        /// @brief Row-major n * n matrix A.
                        const T* matrix_;

                        /// @brief Order of the matrix.
                        std::size_t n_;

                        /// @brief Krylov vectors kept before a restart.
                        std::size_t restart_;

                        /// @brief Order of each diagonal block of the preconditioner.
                        std::size_t block_;

                        /// @brief Every T array of the workspace, laid out by layout_.
                        std::vector<T> values_;

                        /// @brief Row swapped with each row of a block during factoring.
                        std::vector<std::uint32_t> pivots_;

                        /// @brief Where each array starts in values_.
                        GmresLayout layout_;
                };
            } // namespace Detail

            /**
             * @brief Solves A x = b by restarted GMRES with right block-Jacobi preconditioning.
             * @param x Starting guess on entry, the result on exit.
             * @pre Every diagonal block of A is nonsingular; block_size divides n.
             * @see modules/kernels/linalg_kernels/README.md
             */
            template<typename T>
            void gmres(const T* matrix, const T* rhs, T* x, std::uint32_t* iterations, T* residual,
                       std::size_t n, const GmresSettings& settings)
            {
                const GmresLayout layout(n, settings.restart, settings.block_size);
                Detail::GmresSolve<T> solve(matrix, layout);
                solve.solve(rhs, x, iterations, residual, settings);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
