/**************************************************************************/
/* trsm.hpp                                                               */
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
 * @file trsm.hpp
 * @brief Defines the SYCL kernel of TRSM: a blocked substitution with parallel updates.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_system.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Holds what both launches of one diagonal block read: the system, the
                 *        operands and the block's steps.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct TriangularBlock
                {
                    TriangularSystem system;         /**< The decoded substitution. */
                    const T* a = nullptr;            /**< The first A of the batch. */
                    int64_t str_a = 0;               /**< Elements from one A to the next. */
                    T* b = nullptr;                  /**< The first B of the batch. */
                    int64_t str_b = 0;               /**< Elements from one B to the next. */
                    Accumulator<T>* work = nullptr;  /**< The wide solution of every B, or null. */
                    T alpha{};                       /**< The scale of the right-hand side. */
                    int64_t first = 0;               /**< The first step of the block. */
                    int64_t last = 0;                /**< One past the last step of the block. */
                    bool opens = false;              /**< Whether no earlier block has touched B. */

                    /** @brief Returns the A of one batch entry. */
                    [[nodiscard]] const T* matrix(int64_t batch_id) const noexcept
                    {
                        return a + batch_id * str_a;
                    }

                    /** @brief Returns the B of one batch entry. */
                    [[nodiscard]] T* right_hand_side(int64_t batch_id) const noexcept
                    {
                        return b + batch_id * str_b;
                    }

                    /** @brief Returns the running solution of one batch entry. */
                    [[nodiscard]] RunningSolution<T> solution(int64_t batch_id) const noexcept
                    {
                        return RunningSolution<T>(right_hand_side(batch_id), work,
                                                  batch_id * system.order * system.vectors);
                    }

                    /**
                     * @brief Returns the running value of one unknown before this block touches it.
                     * @return alpha * b when the block opens the solve, the stored value otherwise.
                     */
                    [[nodiscard]] Accumulator<T> running(const T* B, const RunningSolution<T>& X,
                                                         int64_t index) const noexcept
                    {
                        using W = Accumulator<T>;
                        return opens ? multiply(W(alpha), W(B[index])) : X.get(index);
                    }
                };

                /**
                 * @brief Solves the steps of one diagonal block for one right-hand-side vector.
                 * @see kernels/README.md
                 */
                template<typename T>
                class TriangularBlockSolve final
                {
                    public:
                        /** @brief Binds the block whose steps are solved. */
                        explicit TriangularBlockSolve(const TriangularBlock<T>& block) noexcept
                            : block_(block)
                        {

                        }

                        /** @brief Solves the block for the vector and the matrix index names. */
                        void operator()(std::size_t index) const
                        {
                            using W = Accumulator<T>;
                            const TriangularSystem& system = block_.system;
                            const int64_t stream = static_cast<int64_t>(index);
                            const int64_t batch_id = stream / system.vectors;
                            const int64_t v = stream % system.vectors;

                            const T* A = block_.matrix(batch_id);
                            const T* B = block_.right_hand_side(batch_id);
                            const RunningSolution<T> X = block_.solution(batch_id);

                            if (system.forward)
                            {
                                for (int64_t s = block_.first; s < block_.last; ++s)
                                {
                                    const int64_t target = system.unknown.index(s, v);
                                    W sum = block_.running(B, X, target);
                                    for (int64_t t = block_.first; t < s; ++t)
                                        sum -= multiply(W(system.coefficient.at(A, s, t)),
                                                        X.get(system.unknown.index(t, v)));
                                    X.set(target,
                                          system.unit_diag ? sum : divide(sum, pivot(A, s)));
                                }
                            }
                            else
                            {
                                for (int64_t s = block_.last - 1; s >= block_.first; --s)
                                {
                                    const int64_t target = system.unknown.index(s, v);
                                    W sum = block_.running(B, X, target);
                                    for (int64_t t = block_.last - 1; t > s; --t)
                                        sum -= multiply(W(system.coefficient.at(A, s, t)),
                                                        X.get(system.unknown.index(t, v)));
                                    X.set(target,
                                          system.unit_diag ? sum : divide(sum, pivot(A, s)));
                                }
                            }
                        }

                    private:
                        /** @brief Returns the diagonal factor of step s, in Accumulator<T>. */
                        [[nodiscard]] Accumulator<T> pivot(const T* A, int64_t s) const noexcept
                        {
                            return Accumulator<T>(block_.system.coefficient.at(A, s, s));
                        }

                        /// @brief The block whose steps are solved.
                        TriangularBlock<T> block_;
                };

                /**
                 * @brief Subtracts one solved diagonal block from the running value of one unknown
                 *        outside it.
                 * @see kernels/README.md
                 */
                template<typename T>
                class TriangularBlockUpdate final
                {
                    public:
                        /**
                         * @brief Binds the solved block and the steps still to be solved.
                         * @param trailing_first The first unsolved step.
                         * @param trailing_count Unsolved steps, at least one.
                         */
                        TriangularBlockUpdate(const TriangularBlock<T>& block,
                                              int64_t trailing_first,
                                              int64_t trailing_count) noexcept
                            : block_(block), trailing_first_(trailing_first),
                              trailing_count_(trailing_count),
                              step_is_innermost_(block.system.unknown.row == 1)
                        {

                        }

                        /** @brief Updates the unsolved unknown that index names. */
                        void operator()(std::size_t index) const
                        {
                            using W = Accumulator<T>;
                            const TriangularSystem& system = block_.system;
                            const int64_t per_matrix = trailing_count_ * system.vectors;
                            const int64_t element = static_cast<int64_t>(index);
                            const int64_t batch_id = element / per_matrix;
                            const int64_t within = element % per_matrix;

                            // The index that moves one element of B runs innermost.
                            const int64_t s = trailing_first_ + (step_is_innermost_
                                                                     ? within % trailing_count_
                                                                     : within / system.vectors);
                            const int64_t v = step_is_innermost_ ? within / trailing_count_
                                                                 : within % system.vectors;

                            const T* A = block_.matrix(batch_id);
                            const T* B = block_.right_hand_side(batch_id);
                            const RunningSolution<T> X = block_.solution(batch_id);
                            const int64_t target = system.unknown.index(s, v);

                            W sum = block_.running(B, X, target);
                            if (system.forward)
                            {
                                for (int64_t t = block_.first; t < block_.last; ++t)
                                    sum -= multiply(W(system.coefficient.at(A, s, t)),
                                                    X.get(system.unknown.index(t, v)));
                            }
                            else
                            {
                                for (int64_t t = block_.last - 1; t >= block_.first; --t)
                                    sum -= multiply(W(system.coefficient.at(A, s, t)),
                                                    X.get(system.unknown.index(t, v)));
                            }
                            X.set(target, sum);
                        }

                    private:
                        /// @brief The solved block whose terms are subtracted.
                        TriangularBlock<T> block_;

                        /// @brief The first unsolved step.
                        int64_t trailing_first_;

                        /// @brief How many steps are unsolved.
                        int64_t trailing_count_;

                        /// @brief Whether consecutive steps of one vector are adjacent in B.
                        bool step_is_innermost_;
                };
            } // namespace Detail

            /**
             * @brief Solves op(A) * X = alpha * B or X * op(A) = alpha * B for X, overwriting B, in
             *        the blocks a plan gives.
             * @param work `batch_size * m * n` elements under K_NEEDS_WORKSPACE, else never read.
             * @pre When work is read, B has no padding: its last element is element m * n - 1.
             * @return The event of the last launch; every launch is ordered behind deps.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event trsm(sycl::queue& queue, const TriangularSolvePlan& plan,
                                           Core::Layout layout, Core::Side side, Core::Uplo uplo,
                                           Core::Transpose trans, Core::Diag diag,
                                           int64_t m, int64_t n, T alpha,
                                           const T* a, int64_t lda, int64_t str_a,
                                           T* b, int64_t ldb, int64_t str_b, int64_t batch_size,
                                           Accumulator<T>* work,
                                           const std::vector<sycl::event>& deps)
            {
                const TriangularSystem system =
                    triangular_system(layout, side, uplo, trans, diag, m, n, lda, ldb);
                const int64_t width = plan.block_width(system.order);
                const int64_t blocks = plan.block_count(system.order);
                const std::size_t streams = static_cast<std::size_t>(batch_size * system.vectors);

                // The workspace is indexed like B, and holds m * n elements per matrix.
                SB_ASSERT(!RunningSolution<T>::K_NEEDS_WORKSPACE || m <= 0 || n <= 0 ||
                              system.unknown.index(system.order - 1, system.vectors - 1) < m * n,
                          "SYCL TRSM workspace cannot follow a padded B");

                SB_LOG_INFO("SYCL TRSM: batch={} m={} n={} block={} blocks={}", batch_size, m, n,
                            width, blocks);

                Detail::TriangularBlock<T> block{system, a, str_a, b, str_b, work, alpha,
                                                 0, 0, false};
                sycl::event previous;

                for (int64_t pass = 0; pass < blocks; ++pass)
                {
                    const int64_t position = system.forward ? pass : blocks - 1 - pass;
                    block.first = position * width;
                    block.last = std::min(system.order, block.first + width);
                    block.opens = (pass == 0);

                    const Detail::TriangularBlockSolve<T> solve(block);
                    const sycl::event solved = queue.submit([&](sycl::handler& h)
                    {
                        if (pass == 0)
                            h.depends_on(deps);
                        else
                            h.depends_on(previous);
                        Launch::indexed(h, queue, streams, solve);
                    });
                    previous = solved;

                    const int64_t trailing_first = system.forward ? block.last : 0;
                    const int64_t trailing_count = system.forward ? system.order - block.last
                                                                  : block.first;
                    if (trailing_count == 0)
                        continue;

                    const Detail::TriangularBlockUpdate<T> update(block, trailing_first,
                                                                  trailing_count);
                    const std::size_t elements =
                        static_cast<std::size_t>(batch_size * trailing_count * system.vectors);
                    previous = queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(solved);
                        Launch::indexed(h, queue, elements, update);
                    });
                }

                return previous;
            }

            /**
             * @brief Solves op(A) * X = alpha * B or X * op(A) = alpha * B for X, overwriting B, in
             *        the blocks the queue's device plans.
             * @param work `batch_size * m * n` elements when RunningSolution<T> needs a workspace;
             *        never read otherwise.
             * @pre When work is read, B has no padding: its last element is element m * n - 1.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event trsm(sycl::queue& queue, Core::Layout layout,
                                           Core::Side side, Core::Uplo uplo, Core::Transpose trans,
                                           Core::Diag diag, int64_t m, int64_t n, T alpha,
                                           const T* a, int64_t lda, int64_t str_a,
                                           T* b, int64_t ldb, int64_t str_b, int64_t batch_size,
                                           Accumulator<T>* work,
                                           const std::vector<sycl::event>& deps)
            {
                const bool left = (side == Core::Side::Left);
                const TriangularSolvePlan plan = plan_triangular_solve(
                    Device::profile_for(queue), left ? m : n, batch_size * (left ? n : m));

                return trsm<T>(queue, plan, layout, side, uplo, trans, diag, m, n, alpha,
                               a, lda, str_a, b, ldb, str_b, batch_size, work, deps);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
