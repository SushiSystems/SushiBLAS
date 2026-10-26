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
 * @brief Defines the SYCL kernel of the restarted, block-Jacobi preconditioned GMRES solve.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/no_contraction.hpp>
#include <SushiBLAS/linalg_kernels/chunked_dot.hpp>
#include <SushiBLAS/linalg_kernels/gmres_workspace.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /** @brief Holds the widest work-group the solve runs in. */
            constexpr std::size_t K_GMRES_GROUP = 256;

            /** @brief Returns the work-group width gmres runs in on the device of @p q. */
            [[nodiscard]] inline std::size_t gmres_group(const sycl::queue& q)
            {
                const std::size_t device_limit =
                    q.get_device().get_info<sycl::info::device::max_work_group_size>();
                return std::min(K_GMRES_GROUP, device_limit);
            }

            /**
             * @brief Solves A x = b by restarted GMRES with right block-Jacobi preconditioning.
             * @param workspace GmresLayout(n, restart, block_size).bytes(sizeof(T)) device bytes.
             * @note One work-group runs the solve; every dot product is chunked_dot's order, so
             *       the result depends on no launch width.
             * @pre Every diagonal block of A is nonsingular; block_size divides n.
             * @see modules/kernels/linalg_kernels/README.md
             */
            template<typename T>
            sycl::event gmres(sycl::queue& q, const T* matrix, const T* rhs, T* x,
                              std::uint32_t* iterations, T* residual, std::size_t n,
                              const GmresSettings& settings, void* workspace,
                              const std::vector<sycl::event>& deps)
            {
                const GmresLayout layout(n, settings.restart, settings.block_size);
                const std::size_t group = gmres_group(q);
                const std::size_t restart = settings.restart;
                const std::size_t block = settings.block_size;
                const std::size_t max_iterations = settings.max_iterations;
                const T tolerance = static_cast<T>(settings.relative_tolerance);
                T* values = static_cast<T*>(workspace);
                auto* pivots = reinterpret_cast<std::uint32_t*>(values + layout.scalars);

                return q.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    sycl::local_accessor<T, 1> shared(sycl::range<1>(1), h);
                    sycl::local_accessor<int, 1> stop(sycl::range<1>(1), h);
                    h.parallel_for(sycl::nd_range<1>(group, group), [=](sycl::nd_item<1> it)
                    {
                        SB_FP_CONTRACT_OFF
                        const sycl::group<1> g = it.get_group();
                        const std::size_t lid = it.get_local_id(0);
                        const std::size_t chunks = dot_chunks(n);
                        T* chunk_sums = values + layout.partials;
                        T* basis = values + layout.basis;
                        T* hessenberg = values + layout.hessenberg;
                        T* cosines = values + layout.cosines;
                        T* sines = values + layout.sines;
                        T* projection = values + layout.projection;
                        T* coefficients = values + layout.coefficients;
                        T* r0 = values + layout.residual;
                        T* work = values + layout.work;
                        T* preconditioned = values + layout.preconditioned;
                        T* factors = values + layout.factors;

                        auto hess = [=](std::size_t row, std::size_t column) -> T&
                        {
                            return hessenberg[row * restart + column];
                        };

                        // Every item calls dot together; it returns the same value to each.
                        auto dot = [&](const T* a, const T* b) -> T
                        {
                            for (std::size_t c = lid; c < chunks; c += group)
                                chunk_sums[c] = chunk_dot(a, b, n, c);
                            sycl::group_barrier(g);
                            if (lid == 0)
                            {
                                T total = T(0);
                                for (std::size_t c = 0; c < chunks; ++c)
                                    total += chunk_sums[c];
                                shared[0] = total;
                            }
                            sycl::group_barrier(g);
                            return shared[0];
                        };

                        auto multiply = [&](const T* in, T* out)
                        {
                            for (std::size_t row = lid; row < n; row += group)
                                out[row] = chunked_dot(matrix + row * n, in, n);
                            sycl::group_barrier(g);
                        };

                        auto precondition = [&](const T* in, T* out)
                        {
                            SB_FP_CONTRACT_OFF
                            for (std::size_t start = lid * block; start < n; start += group * block)
                            {
                                const T* lu = factors + start * block;
                                const std::uint32_t* pivot = pivots + start;
                                T* y = out + start;
                                for (std::size_t r = 0; r < block; ++r)
                                    y[r] = in[start + r];
                                for (std::size_t k = 0; k < block; ++k)
                                    if (pivot[k] != k)
                                    {
                                        const T held = y[k];
                                        y[k] = y[pivot[k]];
                                        y[pivot[k]] = held;
                                    }
                                for (std::size_t r = 1; r < block; ++r)
                                    for (std::size_t c = 0; c < r; ++c)
                                    {
                                        const T product = lu[r * block + c] * y[c];
                                        y[r] -= product;
                                    }
                                for (std::size_t r = block; r-- > 0;)
                                {
                                    for (std::size_t c = r + 1; c < block; ++c)
                                    {
                                        const T product = lu[r * block + c] * y[c];
                                        y[r] -= product;
                                    }
                                    y[r] /= lu[r * block + r];
                                }
                            }
                            sycl::group_barrier(g);
                        };

                        for (std::size_t start = lid * block; start < n; start += group * block)
                        {
                            T* lu = factors + start * block;
                            for (std::size_t r = 0; r < block; ++r)
                                for (std::size_t c = 0; c < block; ++c)
                                    lu[r * block + c] = matrix[(start + r) * n + start + c];
                            std::uint32_t* pivot = pivots + start;
                            for (std::size_t k = 0; k < block; ++k)
                            {
                                std::size_t best = k;
                                for (std::size_t r = k + 1; r < block; ++r)
                                    if (sycl::fabs(lu[r * block + k]) >
                                        sycl::fabs(lu[best * block + k]))
                                        best = r;
                                pivot[k] = static_cast<std::uint32_t>(best);
                                if (best != k)
                                    for (std::size_t c = 0; c < block; ++c)
                                    {
                                        const T held = lu[k * block + c];
                                        lu[k * block + c] = lu[best * block + c];
                                        lu[best * block + c] = held;
                                    }
                                for (std::size_t r = k + 1; r < block; ++r)
                                {
                                    const T factor = lu[r * block + k] / lu[k * block + k];
                                    lu[r * block + k] = factor;
                                    for (std::size_t c = k + 1; c < block; ++c)
                                    {
                                        const T product = factor * lu[k * block + c];
                                        lu[r * block + c] -= product;
                                    }
                                }
                            }
                        }
                        sycl::group_barrier(g);

                        const T rhs_norm = sycl::sqrt(dot(rhs, rhs));
                        if (rhs_norm == T(0))
                        {
                            for (std::size_t i = lid; i < n; i += group)
                                x[i] = T(0);
                            if (lid == 0)
                            {
                                iterations[0] = 0U;
                                residual[0] = T(0);
                            }
                            return;
                        }

                        std::size_t taken = 0;
                        T relative = T(0);
                        for (;;)
                        {
                            multiply(x, work);
                            for (std::size_t i = lid; i < n; i += group)
                                r0[i] = rhs[i] - work[i];
                            sycl::group_barrier(g);
                            const T beta = sycl::sqrt(dot(r0, r0));
                            relative = beta / rhs_norm;
                            if (relative <= tolerance || taken >= max_iterations)
                                break;

                            const std::size_t budget = max_iterations - taken;
                            for (std::size_t i = lid; i < n; i += group)
                                basis[i] = r0[i] / beta;
                            if (lid == 0)
                            {
                                for (std::size_t i = 0; i <= restart; ++i)
                                    projection[i] = T(0);
                                projection[0] = beta;
                            }
                            sycl::group_barrier(g);

                            std::size_t steps = 0;
                            while (steps < restart && steps < budget)
                            {
                                const std::size_t j = steps;
                                T* vj = basis + j * n;
                                precondition(vj, preconditioned);
                                multiply(preconditioned, work);
                                if (lid == 0)
                                    for (std::size_t i = 0; i <= j; ++i)
                                        hess(i, j) = T(0);
                                sycl::group_barrier(g);

                                for (int pass = 0; pass < 2; ++pass)
                                    for (std::size_t i = 0; i <= j; ++i)
                                    {
                                        const T* vi = basis + i * n;
                                        const T coefficient = dot(work, vi);
                                        if (lid == 0)
                                            hess(i, j) += coefficient;
                                        for (std::size_t e = lid; e < n; e += group)
                                        {
                                            const T product = coefficient * vi[e];
                                            work[e] -= product;
                                        }
                                        sycl::group_barrier(g);
                                    }

                                const T next_norm = sycl::sqrt(dot(work, work));
                                if (next_norm != T(0))
                                    for (std::size_t e = lid; e < n; e += group)
                                        basis[(j + 1) * n + e] = work[e] / next_norm;

                                if (lid == 0)
                                {
                                    hess(j + 1, j) = next_norm;
                                    for (std::size_t i = 0; i < j; ++i)
                                    {
                                        const T cos_top = cosines[i] * hess(i, j);
                                        const T sin_bottom = sines[i] * hess(i + 1, j);
                                        const T sin_top = sines[i] * hess(i, j);
                                        const T cos_bottom = cosines[i] * hess(i + 1, j);
                                        hess(i + 1, j) = cos_bottom - sin_top;
                                        hess(i, j) = cos_top + sin_bottom;
                                    }
                                    const T top_square = hess(j, j) * hess(j, j);
                                    const T bottom_square = hess(j + 1, j) * hess(j + 1, j);
                                    const T radius = sycl::sqrt(top_square + bottom_square);
                                    cosines[j] = radius == T(0) ? T(1) : hess(j, j) / radius;
                                    sines[j] = radius == T(0) ? T(0) : hess(j + 1, j) / radius;
                                    hess(j, j) = radius;
                                    hess(j + 1, j) = T(0);
                                    projection[j + 1] = -sines[j] * projection[j];
                                    projection[j] = cosines[j] * projection[j];
                                    stop[0] = (sycl::fabs(projection[j + 1]) / rhs_norm <=
                                                   tolerance ||
                                               next_norm == T(0))
                                                  ? 1 : 0;
                                }
                                sycl::group_barrier(g);
                                ++steps;
                                if (stop[0] != 0)
                                    break;
                            }

                            if (lid == 0)
                                for (std::size_t r = steps; r-- > 0;)
                                {
                                    T value = projection[r];
                                    for (std::size_t c = r + 1; c < steps; ++c)
                                    {
                                        const T product = hess(r, c) * coefficients[c];
                                        value -= product;
                                    }
                                    coefficients[r] =
                                        hess(r, r) == T(0) ? T(0) : value / hess(r, r);
                                }
                            sycl::group_barrier(g);

                            for (std::size_t e = lid; e < n; e += group)
                            {
                                T value = T(0);
                                for (std::size_t k = 0; k < steps; ++k)
                                {
                                    const T product = coefficients[k] * basis[k * n + e];
                                    value += product;
                                }
                                work[e] = value;
                            }
                            sycl::group_barrier(g);
                            precondition(work, preconditioned);
                            for (std::size_t e = lid; e < n; e += group)
                                x[e] += preconditioned[e];
                            sycl::group_barrier(g);
                            taken += steps;
                        }

                        if (lid == 0)
                        {
                            iterations[0] = static_cast<std::uint32_t>(taken);
                            residual[0] = relative;
                        }
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
