/**************************************************************************/
/* orthonormalize.hpp                                                     */
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
 * @file orthonormalize.hpp
 * @brief Defines the work-group parallel Gram-Schmidt sweep behind RandomOps::orthogonal.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/launch/submitted_events.hpp>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Random
            {
                /** @brief Holds the helpers of the orthonormalisation sweep. */
                namespace Detail
                {
                    /** @brief Holds the largest work-group size of the sweep; the tree needs a power of two. */
                    inline constexpr std::size_t K_ORTHO_MAX_GROUP = 64;

                    /** @brief Returns the largest power of two at most K_ORTHO_MAX_GROUP that the device accepts. */
                    inline std::size_t ortho_group_size(const sycl::queue& q)
                    {
                        return Reduce::floor_pow2(
                            std::min(K_ORTHO_MAX_GROUP, Device::profile_for(q).max_work_group_size));
                    }

                    /**
                     * @brief Sums two values across the work-group by a fixed pairwise tree in local memory.
                     * @pre scratch holds 2 * group elements and group is a power of two.
                     */
                    template<typename T>
                    inline void group_tree_sum2(const sycl::nd_item<1>& item, const sycl::local_accessor<T, 1>& scratch,
                                                std::size_t group, T& first, T& second)
                    {
                        const std::size_t lid = item.get_local_id(0);
                        scratch[lid] = first;
                        scratch[group + lid] = second;
                        sycl::group_barrier(item.get_group());

                        for (std::size_t stride = group / 2; stride > 0; stride /= 2)
                        {
                            if (lid < stride)
                            {
                                scratch[lid] += scratch[lid + stride];
                                scratch[group + lid] += scratch[group + lid + stride];
                            }
                            sycl::group_barrier(item.get_group());
                        }

                        first = scratch[0];
                        second = scratch[group];
                    }

                    /**
                     * @brief Projects vector @p i out of every later vector, one work-group per later vector.
                     * @pre i + 1 < set.count.
                     */
                    template<typename T>
                    inline sycl::event project_step(sycl::queue& q, const std::vector<sycl::event>& deps, T* p,
                                                    const VectorSet& set, int64_t i, std::size_t group)
                    {
                        const std::size_t later = static_cast<std::size_t>(set.count - i - 1);

                        return q.submit([&](sycl::handler& h)
                        {
                            h.depends_on(deps);
                            sycl::local_accessor<T, 1> scratch(sycl::range<1>(2 * group), h);
                            const VectorSet s = set;

                            h.parallel_for(sycl::nd_range<1>(later * group, group), [=](sycl::nd_item<1> item)
                            {
                                const int64_t lid = static_cast<int64_t>(item.get_local_id(0));
                                const int64_t step = static_cast<int64_t>(group);
                                const T* vi = p + i * s.vec_stride;
                                T* vm = p + (i + 1 + static_cast<int64_t>(item.get_group(0))) * s.vec_stride;

                                T norm_sq = static_cast<T>(0);
                                T dot = static_cast<T>(0);
                                for (int64_t j = lid; j < s.length; j += step)
                                {
                                    const T a = vi[j * s.elem_stride];
                                    norm_sq += a * a;
                                    dot += a * vm[j * s.elem_stride];
                                }
                                group_tree_sum2(item, scratch, group, norm_sq, dot);

                                const T coef = norm_sq > tiny_norm_sq<T>() ? dot / norm_sq : static_cast<T>(0);
                                for (int64_t j = lid; j < s.length; j += step)
                                {
                                    vm[j * s.elem_stride] -= coef * vi[j * s.elem_stride];
                                }
                            });
                        });
                    }

                    /**
                     * @brief Scales every vector to norm @p gain, one work-group per vector.
                     * @see kernels/README.md
                     */
                    template<typename T>
                    inline sycl::event normalize_step(sycl::queue& q, const std::vector<sycl::event>& deps, T* p,
                                                      const VectorSet& set, T gain, std::size_t group)
                    {
                        return q.submit([&](sycl::handler& h)
                        {
                            h.depends_on(deps);
                            sycl::local_accessor<T, 1> scratch(sycl::range<1>(2 * group), h);
                            const VectorSet s = set;

                            h.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(s.count) * group, group),
                                           [=](sycl::nd_item<1> item)
                            {
                                const int64_t lid = static_cast<int64_t>(item.get_local_id(0));
                                const int64_t step = static_cast<int64_t>(group);
                                T* v = p + static_cast<int64_t>(item.get_group(0)) * s.vec_stride;

                                T norm_sq = static_cast<T>(0);
                                T unused = static_cast<T>(0);
                                for (int64_t j = lid; j < s.length; j += step)
                                {
                                    const T a = v[j * s.elem_stride];
                                    norm_sq += a * a;
                                }
                                group_tree_sum2(item, scratch, group, norm_sq, unused);

                                const T scale = norm_sq > tiny_norm_sq<T>() ? gain / sycl::sqrt(norm_sq) : gain;
                                for (int64_t j = lid; j < s.length; j += step)
                                {
                                    v[j * s.elem_stride] *= scale;
                                }
                            });
                        });
                    }
                } // namespace Detail

                /**
                 * @brief Orthonormalises a strided set of vectors in place and scales them by @p gain.
                 * @tparam T The real element type, float or double.
                 * @return The events of every submitted kernel, in order.
                 * @see kernels/README.md
                 */
                template<typename T>
                [[nodiscard]] inline Device::SubmittedEvents orthonormalize(sycl::queue& q, const std::vector<sycl::event>& deps,
                                                                            T* p, const VectorSet& set, T gain)
                {
                    const std::size_t group = Detail::ortho_group_size(q);
                    std::vector<sycl::event> chain = deps;
                    Device::SubmittedEvents submitted;

                    for (int sweep = 0; sweep < K_ORTHONORMAL_SWEEPS; ++sweep)
                    {
                        for (int64_t i = 0; i + 1 < set.count; ++i)
                        {
                            chain = {Detail::project_step(q, chain, p, set, i, group)};
                            submitted.append(chain.front());
                        }
                    }

                    submitted.append(Detail::normalize_step(q, chain, p, set, gain, group));
                    return submitted;
                }

            } // namespace Random
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
