/**************************************************************************/
/* reduce_axis.hpp                                                        */
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
 * @file reduce_axis.hpp
 * @brief Declares the segmented reduction: many independent folds, one result each.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/reduce_1d.hpp>
#include <SushiBLAS/engine/device_policy.hpp>

namespace SushiBLAS
{
    namespace Reduce
    {
        /**
         * @brief Holds where one segmented reduction's inputs and outputs live, in element counts.
         * @see include/SushiBLAS/engine/README.md
         */
        struct AxisSpan
        {
            /** @brief How many independent folds; equals the result's length. */
            std::size_t segments = 0;

            /** @brief How many elements are folded into each segment's result. */
            std::size_t extent = 0;

            /** @brief Input elements between the starts of consecutive segments. */
            std::size_t segment_stride = 0;

            /** @brief Input elements between consecutive elements of one segment. */
            std::size_t element_stride = 1;

            /** @brief Output elements between consecutive results. */
            std::size_t out_stride = 1;

            /**
             * @brief Holds the number of groups of segments; 1 when there is one segment axis.
             * @see include/SushiBLAS/engine/README.md
             */
            std::size_t segments2 = 1;

            /** @brief Input elements between the starts of consecutive segment groups. */
            std::size_t segment_stride2 = 0;

            /** @brief How many folds in total: `segments * segments2`. */
            [[nodiscard]] std::size_t total_segments() const noexcept
            {
                return segments * (segments2 == 0 ? 0 : segments2);
            }
        };

        /**
         * @struct AxisPlan
         * @brief The launch geometry of one segmented reduction.
         */
        struct AxisPlan
        {
            /**
             * @brief Work-items cooperating on a single segment; 1 for none.
             *
             * Always a power of two when it exceeds one, because the halving tree
             * only visits every lane when it is - the same requirement
             * @ref floor_pow2 exists to satisfy in `reduce_1d.hpp`.
             */
            std::size_t local = 1;

            /** @brief True when a whole work-group folds one segment together. */
            [[nodiscard]] bool cooperative() const noexcept { return local > 1; }
        };

        /**
         * @brief Picks between the two segmented shapes for segments folds of extent elements on p.
         * @return The geometry; AxisPlan::cooperative says which shape.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline AxisPlan plan_axis(const Device::Profile& p,
                                                std::size_t segments,
                                                std::size_t extent) noexcept
        {
            if (segments == 0 || extent == 0) return AxisPlan{1};

            const std::size_t resident = p.is_cpu
                ? p.compute_units
                : p.compute_units * Device::K_WAVES_PER_COMPUTE_UNIT;

            if (segments >= resident) return AxisPlan{1};

            const std::size_t local = group_width(p);
            if (local < 2 || extent < local) return AxisPlan{1};

            return AxisPlan{local};
        }

        /**
         * @brief Reduces each segment of span to one element of out, in a fixed order.
         * @param geometry Which shape to launch, normally plan_axis's answer.
         * @param transform Callable `(T) -> T`, applied to each element before the fold.
         * @param finalize Callable `(T value, std::size_t n) -> T`, applied once per segment.
         * @return The event for the submitted kernel.
         * @see include/SushiBLAS/engine/README.md
         */
        template<template<typename> class Accum, typename T,
                 typename Transform, typename Finalize>
        [[nodiscard]] sycl::event reduce_axis(sycl::queue& queue, const T* in, T* out,
                                              const AxisSpan& span, const AxisPlan& geometry,
                                              Transform transform, Finalize finalize,
                                              const std::vector<sycl::event>& deps)
        {
            const std::size_t segments = span.segments;
            const std::size_t extent = span.extent;
            const std::size_t segment_stride = span.segment_stride;
            const std::size_t element_stride = span.element_stride;
            const std::size_t out_stride = span.out_stride;
            const std::size_t segment_stride2 = span.segment_stride2;
            const std::size_t total = span.total_segments();

            if (!geometry.cooperative())
            {
                // One work-item folds one whole segment; see include/SushiBLAS/engine/README.md.
                const Device::Grid1D grid = Device::plan_1d(Device::profile_for(queue), total);
                const std::size_t stride = grid.stride();

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t s = it.get_global_id(0); s < total; s += stride)
                        {
                            // The segment is folded in ascending index order and no other work-item touches it; see include/SushiBLAS/engine/README.md.
                            Accum<T> acc;
                            // One division per fold; a segments2 == 1 kernel is not worth a second instantiation, see include/SushiBLAS/engine/README.md.
                            const std::size_t s2 = s / segments;
                            const std::size_t s1 = s % segments;
                            const T* base = in + s2 * segment_stride2 + s1 * segment_stride;
                            for (std::size_t k = 0; k < extent; ++k)
                                acc.accumulate(transform(base[k * element_stride]));

                            out[s * out_stride] = finalize(acc.value(), extent);
                        }
                    });
                });
            }

            const std::size_t local = geometry.local;

            return queue.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);

                sycl::local_accessor<T, 1> tile(sycl::range<1>(local), h);

                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(total * local),
                                                 sycl::range<1>(local)),
                               [=](sycl::nd_item<1> it)
                {
                    const std::size_t s = it.get_group(0);
                    const std::size_t lid = it.get_local_id(0);

                    // One division per fold; a segments2 == 1 kernel is not worth a second instantiation, see include/SushiBLAS/engine/README.md.
                    const std::size_t s2 = s / segments;
                    const std::size_t s1 = s % segments;
                    const T* base = in + s2 * segment_stride2 + s1 * segment_stride;

                    // Lane l owns positions l, l + local, l + 2*local, fixed by the geometry; see include/SushiBLAS/engine/README.md.
                    Accum<T> acc;
                    for (std::size_t k = lid; k < extent; k += local)
                        acc.accumulate(transform(base[k * element_stride]));

                    tile[lid] = acc.value();

                    sycl::group_barrier(it.get_group());

                    // A fixed-order halving tree; local is a power of two; see include/SushiBLAS/engine/README.md.
                    for (std::size_t off = local / 2; off > 0; off >>= 1)
                    {
                        if (lid < off) tile[lid] = Accum<T>::combine(tile[lid], tile[lid + off]);
                        sycl::group_barrier(it.get_group());
                    }

                    if (lid == 0) out[s * out_stride] = finalize(tile[0], extent);
                });
            });
        }

    } // namespace Reduce
} // namespace SushiBLAS
