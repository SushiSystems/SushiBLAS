/**************************************************************************/
/* triangular_solve.hpp                                                   */
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
 * @file triangular_solve.hpp
 * @brief Defines the SYCL triangular substitution TBSV and TPSV share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/row_cursor.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Holds what every launch of a one-vector solve reads: the triangle, the
                 *        vector and the rows of one diagonal block.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                struct VectorSolveBlock
                {
                    Access access;                  /**< How op(A) is read. */
                    const T* a = nullptr;           /**< The stored triangle. */
                    T* x = nullptr;                 /**< The right-hand side, then the solution. */
                    int64_t incx = 1;               /**< Distance between elements of x. */
                    Accumulator<T>* work = nullptr; /**< The wide solution, or null. */
                    int64_t n = 0;                  /**< Rows and columns of op(A). */
                    int64_t first = 0;              /**< The first row of the block. */
                    int64_t last = 0;               /**< One past the last row of the block. */
                    bool forward = true;            /**< Whether row i needs the rows above it. */
                    bool unit_diag = false;         /**< Whether the diagonal is one, never read. */

                    /** @brief Returns the running solution, addressed like x. */
                    [[nodiscard]] RunningSolution<T> solution() const noexcept
                    {
                        return RunningSolution<T>(x, work, 0);
                    }

                    /** @brief Returns the first column of row i that lies in the block. */
                    [[nodiscard]] int64_t first_in_block(int64_t i) const noexcept
                    {
                        return std::max(first, access.first_column(i));
                    }

                    /** @brief Returns one past the last column of row i that lies in the block. */
                    [[nodiscard]] int64_t end_in_block(int64_t i) const noexcept
                    {
                        return std::min(last, access.end_column(i, n));
                    }

                    /**
                     * @brief Returns sum less the terms of columns [from, to) of row i, taken
                     *        in ascending column order.
                     */
                    [[nodiscard]] Accumulator<T> less_ascending(const RunningSolution<T>& X,
                                                                Accumulator<T> sum, int64_t i,
                                                                int64_t from,
                                                                int64_t to) const noexcept
                    {
                        using W = Accumulator<T>;

                        RowCursor cursor = access.row_cursor(i, from);
                        for (int64_t t = from; t < to; ++t, cursor.advance())
                            sum -= multiply(W(access.at(a, cursor)), X.get(t * incx));
                        return sum;
                    }

                    /**
                     * @brief Returns sum less the terms of columns [from, to) of row i, taken
                     *        in descending column order.
                     */
                    [[nodiscard]] Accumulator<T> less_descending(const RunningSolution<T>& X,
                                                                 Accumulator<T> sum, int64_t i,
                                                                 int64_t from,
                                                                 int64_t to) const noexcept
                    {
                        using W = Accumulator<T>;

                        RowCursor cursor = access.row_cursor(i, to - 1);
                        for (int64_t t = to - 1; t >= from; --t, cursor.retreat())
                            sum -= multiply(W(access.at(a, cursor)), X.get(t * incx));
                        return sum;
                    }

                    /** @brief Returns the running value of row i divided by its diagonal. */
                    [[nodiscard]] Accumulator<T> solved(Accumulator<T> sum,
                                                        int64_t i) const noexcept
                    {
                        using W = Accumulator<T>;
                        return unit_diag ? sum : divide(sum, W(access.at(a, i, i)));
                    }
                };

                /**
                 * @brief Copies element i of the right-hand side into the running solution.
                 * @see kernels/README.md
                 */
                template<typename T>
                struct VectorSolveOpen
                {
                    T* x = nullptr;                 /**< The right-hand side. */
                    int64_t incx = 1;               /**< Distance between elements of x. */
                    Accumulator<T>* work = nullptr; /**< The wide solution. */

                    /** @brief Opens element index. */
                    void operator()(std::size_t index) const
                    {
                        const int64_t at = static_cast<int64_t>(index) * incx;
                        RunningSolution<T>(x, work, 0).set(at, Accumulator<T>(x[at]));
                    }
                };

                /**
                 * @brief Solves the rows of one diagonal block in order, in one work-item.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                struct VectorSolveSteps
                {
                    VectorSolveBlock<T, Access> block; /**< The block whose rows are solved. */

                    /** @brief Solves every row of the block. */
                    void operator()() const
                    {
                        const RunningSolution<T> X = block.solution();

                        if (block.forward)
                        {
                            for (int64_t i = block.first; i < block.last; ++i)
                            {
                                const Accumulator<T> sum = block.less_ascending(
                                    X, X.get(i * block.incx), i, block.first_in_block(i), i);
                                X.set(i * block.incx, block.solved(sum, i));
                            }
                        }
                        else
                        {
                            for (int64_t i = block.last - 1; i >= block.first; --i)
                            {
                                const Accumulator<T> sum = block.less_descending(
                                    X, X.get(i * block.incx), i, i + 1, block.end_in_block(i));
                                X.set(i * block.incx, block.solved(sum, i));
                            }
                        }
                    }
                };

                /**
                 * @brief Subtracts one solved diagonal block from the running value of one row
                 *        outside it.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                struct VectorSolveUpdate
                {
                    VectorSolveBlock<T, Access> block; /**< The solved block. */
                    int64_t trailing_first = 0;        /**< The first row updated. */

                    /** @brief Updates row trailing_first + index. */
                    void operator()(std::size_t index) const
                    {
                        const RunningSolution<T> X = block.solution();
                        const int64_t i = trailing_first + static_cast<int64_t>(index);

                        const Accumulator<T> running = X.get(i * block.incx);
                        X.set(i * block.incx,
                              block.forward
                                  ? block.less_ascending(X, running, i, block.first_in_block(i),
                                                         block.last)
                                  : block.less_descending(X, running, i, block.first,
                                                          block.end_in_block(i)));
                    }
                };

                /**
                 * @brief Solves op(A) * x = b in place, x holding b on entry, in a plan's blocks.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @param reach Later rows one solved row is subtracted from, at most.
                 * @param work (n - 1) * incx + 1 elements if RunningSolution<T> needs them.
                 * @return The event of the last launch; every launch is ordered behind deps.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                [[nodiscard]] sycl::event triangular_solve(sycl::queue& queue,
                                                           const TriangularSolvePlan& plan,
                                                           const Access& access, bool op_upper,
                                                           bool unit_diag, int64_t n, int64_t reach,
                                                           const T* a, T* x, int64_t incx,
                                                           Accumulator<T>* work,
                                                           const std::vector<sycl::event>& deps)
                {
                    const int64_t width = plan.block_width(n);
                    const int64_t blocks = plan.block_count(n);
                    const bool forward = !op_upper;

                    VectorSolveBlock<T, Access> block{access, a, x, incx, work, n, 0, 0, forward,
                                                      unit_diag};
                    std::vector<sycl::event> previous = deps;

                    if (RunningSolution<T>::K_NEEDS_WORKSPACE)
                    {
                        const VectorSolveOpen<T> open{x, incx, work};
                        const std::size_t rows = static_cast<std::size_t>(n);
                        const sycl::event opened = queue.submit([&](sycl::handler& h)
                        {
                            h.depends_on(deps);
                            Launch::indexed(h, queue, rows, open);
                        });
                        previous.assign(1, opened);
                    }

                    sycl::event last;
                    for (int64_t pass = 0; pass < blocks; ++pass)
                    {
                        const int64_t position = forward ? pass : blocks - 1 - pass;
                        block.first = position * width;
                        block.last = std::min(n, block.first + width);

                        const VectorSolveSteps<T, Access> steps{block};
                        last = queue.submit([&](sycl::handler& h)
                        {
                            h.depends_on(previous);
                            h.single_task(steps);
                        });
                        previous.assign(1, last);

                        const int64_t remaining = forward ? n - block.last : block.first;
                        const int64_t trailing = std::min(remaining, reach);
                        if (trailing <= 0)
                            continue;

                        const VectorSolveUpdate<T, Access> update{
                            block, forward ? block.last : block.first - trailing};
                        const std::size_t rows = static_cast<std::size_t>(trailing);
                        last = queue.submit([&](sycl::handler& h)
                        {
                            h.depends_on(previous);
                            Launch::indexed(h, queue, rows, update);
                        });
                        previous.assign(1, last);
                    }

                    return last;
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
