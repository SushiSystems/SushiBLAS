/**************************************************************************/
/* level1.hpp                                                             */
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
 * @file level1.hpp
 * @brief Defines the SYCL kernels of the sparse-dense level 1 routines.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/level1.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes y[p_k] = alpha * x[k] + y[p_k], one work-item per stored entry.
             * @pre No position is stored twice, so no two work-items write one element.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event axpyi(sycl::queue& queue, T alpha, const T* x,
                                            IndexList<Index> positions, T* y, int64_t incy,
                                            const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(positions.count),
                                    [=](std::size_t k)
                    {
                        const int64_t target = positions.at(static_cast<int64_t>(k)) * incy;
                        y[target] = static_cast<T>(W(y[target]) + multiply(W(alpha), W(x[k])));
                    });
                });
            }

            /**
             * @brief Computes sum(Conj(x[k]) * y[p_k]) through a compensated sum.
             * @tparam Conj Unconjugated for doti and dotui, Conjugated for dotci.
             * @param scratch Device scratch for the partials; may be null when scratch_bytes is 0.
             * @note The result may differ in the last bits from a serial left-to-right sum.
             * @see kernels/README.md
             */
            template<typename T, typename Index, typename Conj = Unconjugated>
            [[nodiscard]] sycl::event doti(sycl::queue& queue, const T* x,
                                           IndexList<Index> positions, const T* y, int64_t incy,
                                           T* result, void* scratch, std::size_t scratch_bytes,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Reduce::AccumulationOf<T>;

                const auto term = [=](std::size_t k)
                {
                    const int64_t entry = static_cast<int64_t>(k);
                    return multiply(Conj::apply(static_cast<W>(x[entry])),
                                    static_cast<W>(y[positions.at(entry) * incy]));
                };
                const auto finish = [](W v, std::size_t) { return v; };
                const std::size_t count = static_cast<std::size_t>(positions.count);

                if constexpr (Core::is_complex_v<T>)
                    return Reduce::reduce_indexed<Reduce::CompensatedComplexSum>(
                        queue, count, term, result, static_cast<W*>(scratch),
                        partial_capacity<W>(scratch_bytes), finish, deps);
                else
                    return Reduce::reduce_indexed<Reduce::CompensatedSum>(
                        queue, count, term, result, static_cast<W*>(scratch),
                        partial_capacity<W>(scratch_bytes), finish, deps);
            }

            /**
             * @brief Copies y[p_k] into x[k], one work-item per stored entry.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event gthr(sycl::queue& queue, const T* y, int64_t incy,
                                           IndexList<Index> positions, T* x,
                                           const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(positions.count),
                                    [=](std::size_t k)
                    {
                        x[k] = y[positions.at(static_cast<int64_t>(k)) * incy];
                    });
                });
            }

            /**
             * @brief Copies y[p_k] into x[k] and sets y[p_k] to zero, one work-item per entry.
             * @pre No position is stored twice, so no two work-items write one element.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event gthrz(sycl::queue& queue, T* y, int64_t incy,
                                            IndexList<Index> positions, T* x,
                                            const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(positions.count),
                                    [=](std::size_t k)
                    {
                        const int64_t source = positions.at(static_cast<int64_t>(k)) * incy;
                        x[k] = y[source];
                        y[source] = T(0);
                    });
                });
            }

            /**
             * @brief Copies x[k] into y[p_k], one work-item per stored entry.
             * @pre No position is stored twice, so no two work-items write one element.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event sctr(sycl::queue& queue, const T* x,
                                           IndexList<Index> positions, T* y, int64_t incy,
                                           const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(positions.count),
                                    [=](std::size_t k)
                    {
                        y[positions.at(static_cast<int64_t>(k)) * incy] = x[k];
                    });
                });
            }

            /**
             * @brief Applies the plane rotation of roti, one work-item per stored entry.
             * @pre No position is stored twice, so no two work-items write one element.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event roti(sycl::queue& queue, T* x, IndexList<Index> positions,
                                           T* y, int64_t incy, RealType<T> c, RealType<T> s,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                using WR = Accumulator<RealType<T>>;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(positions.count),
                                    [=](std::size_t k)
                    {
                        const int64_t target = positions.at(static_cast<int64_t>(k)) * incy;
                        const W xk = W(x[k]);
                        const W yk = W(y[target]);
                        x[k] = static_cast<T>(multiply(WR(c), xk) + multiply(WR(s), yk));
                        y[target] = static_cast<T>(multiply(WR(c), yk) - multiply(WR(s), xk));
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
