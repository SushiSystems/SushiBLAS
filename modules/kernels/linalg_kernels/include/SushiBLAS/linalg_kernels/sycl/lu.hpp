/**************************************************************************/
/* lu.hpp                                                                 */
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
 * @file lu.hpp
 * @brief Defines the SYCL kernels of the batched dense LU factorisation and its solve.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/no_contraction.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /** @brief Holds the widest work-group one matrix of lu_factor is given. */
            constexpr std::size_t K_LU_GROUP = 128;

            /**
             * @brief Factors P A = L U with partial pivoting for each of @p batch matrices.
             * @note One work-group per matrix; every element takes its updates in ascending step
             *       order, so the result does not depend on the group's width.
             * @see modules/kernels/linalg_kernels/README.md
             */
            template<typename T>
            sycl::event lu_factor(sycl::queue& q, T* matrices, std::uint32_t* pivots,
                                  std::uint32_t* singular, std::size_t n, std::size_t batch,
                                  const std::vector<sycl::event>& deps)
            {
                if (batch == 0)
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.single_task([] {});
                    });

                const std::size_t device_limit =
                    q.get_device().get_info<sycl::info::device::max_work_group_size>();
                const std::size_t group = std::min(K_LU_GROUP, device_limit);
                const T epsilon = std::numeric_limits<T>::epsilon();

                return q.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    sycl::local_accessor<std::size_t, 1> pivot_row(sycl::range<1>(1), h);
                    sycl::local_accessor<int, 1> failed(sycl::range<1>(1), h);
                    h.parallel_for(sycl::nd_range<1>(batch * group, group),
                        [=](sycl::nd_item<1> it)
                        {
                            SB_FP_CONTRACT_OFF
                            const sycl::group<1> g = it.get_group();
                            const std::size_t b = it.get_group(0);
                            const std::size_t lid = it.get_local_id(0);
                            T* a = matrices + b * n * n;
                            std::uint32_t* p = pivots + b * n;

                            T local = T(0);
                            for (std::size_t e = lid; e < n * n; e += group)
                                local = sycl::fmax(local, sycl::fabs(a[e]));
                            const T largest = sycl::reduce_over_group(g, local, sycl::maximum<T>());
                            const T threshold = static_cast<T>(n) * epsilon * largest;

                            for (std::size_t k = 0; k < n; ++k)
                            {
                                if (lid == 0)
                                {
                                    std::size_t row = k;
                                    T magnitude = sycl::fabs(a[k * n + k]);
                                    for (std::size_t i = k + 1; i < n; ++i)
                                        if (sycl::fabs(a[i * n + k]) > magnitude)
                                        {
                                            row = i;
                                            magnitude = sycl::fabs(a[i * n + k]);
                                        }
                                    pivot_row[0] = row;
                                    failed[0] = magnitude > threshold ? 0 : 1;
                                }
                                sycl::group_barrier(g);

                                if (failed[0] != 0)
                                {
                                    for (std::size_t i = k + lid; i < n; i += group)
                                        p[i] = static_cast<std::uint32_t>(i);
                                    if (lid == 0)
                                        singular[b] = 1U;
                                    return;
                                }

                                const std::size_t row = pivot_row[0];
                                if (lid == 0)
                                    p[k] = static_cast<std::uint32_t>(row);
                                if (row != k)
                                    for (std::size_t j = lid; j < n; j += group)
                                    {
                                        const T held = a[k * n + j];
                                        a[k * n + j] = a[row * n + j];
                                        a[row * n + j] = held;
                                    }
                                sycl::group_barrier(g);

                                const T inverse = T(1) / a[k * n + k];
                                for (std::size_t i = k + 1 + lid; i < n; i += group)
                                    a[i * n + k] *= inverse;
                                sycl::group_barrier(g);

                                const std::size_t width = n - k - 1;
                                for (std::size_t e = lid; e < width * width; e += group)
                                {
                                    const std::size_t i = k + 1 + e / width;
                                    const std::size_t j = k + 1 + e % width;
                                    const T product = a[i * n + k] * a[k * n + j];
                                    a[i * n + j] -= product;
                                }
                                sycl::group_barrier(g);
                            }
                            if (lid == 0)
                                singular[b] = 0U;
                        });
                });
            }

            /**
             * @brief Solves A X = B for each of @p batch matrices lu_factor factored, in place.
             * @note One work-item per right-hand side column, in the reference's order.
             * @pre No matrix of the batch was reported singular.
             * @see modules/kernels/linalg_kernels/README.md
             */
            template<typename T>
            sycl::event lu_solve(sycl::queue& q, const T* factors, const std::uint32_t* pivots,
                                 T* rhs, std::size_t n, std::size_t rhs_count, std::size_t batch,
                                 const std::vector<sycl::event>& deps)
            {
                const std::size_t count = batch * rhs_count;
                if (count == 0 || n == 0)
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.single_task([] {});
                    });

                const Device::Grid1D grid = Device::plan_1d(Device::profile_for(q), count);
                const std::size_t stride = grid.stride();
                return q.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        SB_FP_CONTRACT_OFF
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const std::size_t b = flat / rhs_count;
                            const std::size_t j = flat % rhs_count;
                            const T* lu = factors + b * n * n;
                            const std::uint32_t* p = pivots + b * n;
                            T* x = rhs + b * n * rhs_count + j;

                            for (std::size_t i = 0; i < n; ++i)
                                if (p[i] != i)
                                {
                                    const T held = x[i * rhs_count];
                                    x[i * rhs_count] = x[p[i] * rhs_count];
                                    x[p[i] * rhs_count] = held;
                                }
                            for (std::size_t i = 1; i < n; ++i)
                                for (std::size_t k = 0; k < i; ++k)
                                {
                                    const T product = lu[i * n + k] * x[k * rhs_count];
                                    x[i * rhs_count] -= product;
                                }
                            for (std::size_t i = n; i-- > 0;)
                            {
                                for (std::size_t k = i + 1; k < n; ++k)
                                {
                                    const T product = lu[i * n + k] * x[k * rhs_count];
                                    x[i * rhs_count] -= product;
                                }
                                x[i * rhs_count] *= T(1) / lu[i * n + i];
                            }
                        }
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
