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
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/device_policy.hpp>

namespace SushiBLAS
{
    namespace Reduce
    {
        /**
         * @brief Holds where one segmented reduction's inputs and outputs live, in element counts.
         * @see modules/foundation/support/README.md
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
             * @see modules/foundation/support/README.md
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
         * @see modules/foundation/support/README.md
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
         * @brief Reduces each of segments index ranges `[0, extent)` to one value, in a fixed order.
         * @tparam W The type a term is read in and folded in by Accum.
         * @param geometry Which shape to launch, normally plan_axis's answer.
         * @param open Callable `(std::size_t segment) -> Cursor`; a Cursor is callable `(std::size_t k) -> W`.
         * @param store Callable `(std::size_t segment, W value)`, called once per segment.
         * @see modules/foundation/support/README.md
         */
        template<template<typename> class Accum, typename W, typename Open, typename Store>
        [[nodiscard]] sycl::event reduce_segments(sycl::queue& queue, std::size_t segments,
                                                  std::size_t extent, const AxisPlan& geometry,
                                                  Open open, Store store,
                                                  const std::vector<sycl::event>& deps)
        {
            if (!geometry.cooperative())
            {
                // One work-item folds one whole segment; see modules/foundation/support/README.md.
                const Device::Grid1D grid = Device::plan_1d(Device::profile_for(queue), segments);
                const std::size_t stride = grid.stride();

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t s = it.get_global_id(0); s < segments; s += stride)
                        {
                            // The segment is folded in ascending index order and no other work-item touches it; see modules/foundation/support/README.md.
                            const auto cursor = open(s);
                            Accum<W> acc;
                            for (std::size_t k = 0; k < extent; ++k)
                                acc.accumulate(cursor(k));

                            store(s, acc.value());
                        }
                    });
                });
            }

            const std::size_t local = geometry.local;

            return queue.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);

                sycl::local_accessor<W, 1> tile(sycl::range<1>(local), h);

                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(segments * local),
                                                 sycl::range<1>(local)),
                               [=](sycl::nd_item<1> it)
                {
                    const std::size_t s = it.get_group(0);
                    const std::size_t lid = it.get_local_id(0);
                    const auto cursor = open(s);

                    // Lane l owns positions l, l + local, l + 2*local, fixed by the geometry; see modules/foundation/support/README.md.
                    Accum<W> acc;
                    for (std::size_t k = lid; k < extent; k += local)
                        acc.accumulate(cursor(k));

                    tile[lid] = acc.value();

                    sycl::group_barrier(it.get_group());

                    // A fixed-order halving tree; local is a power of two; see modules/foundation/support/README.md.
                    for (std::size_t off = local / 2; off > 0; off >>= 1)
                    {
                        if (lid < off) tile[lid] = Accum<W>::combine(tile[lid], tile[lid + off]);
                        sycl::group_barrier(it.get_group());
                    }

                    if (lid == 0) store(s, tile[0]);
                });
            });
        }

        namespace Detail
        {
            /**
             * @brief Reads the elements of one segment of an AxisSpan, transformed and widened to W.
             * @see modules/foundation/support/README.md
             */
            template<typename T, typename W, typename Transform>
            class AxisCursor final
            {
                public:
                    /** @brief Binds the cursor to a segment's first element. */
                    AxisCursor(const T* base, std::size_t element_stride, const Transform& transform)
                        : base_(base), element_stride_(element_stride), transform_(transform)
                    {

                    }

                    /** @brief Returns element k of the segment after the transform. */
                    [[nodiscard]] W operator()(std::size_t k) const
                    {
                        return static_cast<W>(transform_(base_[k * element_stride_]));
                    }

                private:
                    /// @brief The segment's first element.
                    const T* base_;

                    /// @brief Input elements between consecutive elements of the segment.
                    std::size_t element_stride_;

                    /// @brief The callable applied to each element before the fold.
                    Transform transform_;
            };

            /**
             * @brief Opens the segments of an AxisSpan for reduce_segments.
             * @see modules/foundation/support/README.md
             */
            template<typename T, typename W, typename Transform>
            class AxisOpen final
            {
                public:
                    /** @brief Binds the opener to an input array and its segment geometry. */
                    AxisOpen(const T* in, const AxisSpan& span, const Transform& transform)
                        : in_(in), segments_(span.segments), segment_stride_(span.segment_stride),
                          segment_stride2_(span.segment_stride2),
                          element_stride_(span.element_stride), transform_(transform)
                    {

                    }

                    /** @brief Returns the cursor of segment s. */
                    [[nodiscard]] AxisCursor<T, W, Transform> operator()(std::size_t s) const
                    {
                        // One division per fold; see modules/foundation/support/README.md.
                        const std::size_t s2 = s / segments_;
                        const std::size_t s1 = s % segments_;
                        return AxisCursor<T, W, Transform>(
                            in_ + s2 * segment_stride2_ + s1 * segment_stride_, element_stride_,
                            transform_);
                    }

                private:
                    /// @brief The input's first element.
                    const T* in_;

                    /// @brief Segments per segment group.
                    std::size_t segments_;

                    /// @brief Input elements between the starts of consecutive segments.
                    std::size_t segment_stride_;

                    /// @brief Input elements between the starts of consecutive segment groups.
                    std::size_t segment_stride2_;

                    /// @brief Input elements between consecutive elements of one segment.
                    std::size_t element_stride_;

                    /// @brief The callable applied to each element before the fold.
                    Transform transform_;
            };

            /**
             * @brief Finalizes a segment's fold and stores it as T.
             * @see modules/foundation/support/README.md
             */
            template<typename T, typename W, typename Finalize>
            class AxisStore final
            {
                public:
                    /** @brief Binds the store to an output array and the segments' extent. */
                    AxisStore(T* out, std::size_t out_stride, std::size_t extent,
                              const Finalize& finalize)
                        : out_(out), out_stride_(out_stride), extent_(extent), finalize_(finalize)
                    {

                    }

                    /** @brief Writes `finalize(value, extent)` to the element of segment s. */
                    void operator()(std::size_t s, W value) const
                    {
                        out_[s * out_stride_] = static_cast<T>(finalize_(value, extent_));
                    }

                private:
                    /// @brief The output's first element.
                    T* out_;

                    /// @brief Output elements between consecutive results.
                    std::size_t out_stride_;

                    /// @brief How many elements each segment folded.
                    std::size_t extent_;

                    /// @brief The callable applied once to each segment's fold.
                    Finalize finalize_;
            };
        } // namespace Detail

        /**
         * @brief Reduces each segment of span to one element of out, in a fixed order, folding in AccumulationOf<T>.
         * @param geometry Which shape to launch, normally plan_axis's answer.
         * @param transform Callable `(T) -> auto`, applied to each element before the fold.
         * @param finalize Callable `(AccumulationOf<T> value, std::size_t n) -> auto`, applied once per segment.
         * @return The event for the submitted kernel.
         * @see modules/foundation/support/README.md
         */
        template<template<typename> class Accum, typename T,
                 typename Transform, typename Finalize>
        [[nodiscard]] sycl::event reduce_axis(sycl::queue& queue, const T* in, T* out,
                                              const AxisSpan& span, const AxisPlan& geometry,
                                              Transform transform, Finalize finalize,
                                              const std::vector<sycl::event>& deps)
        {
            using W = AccumulationOf<T>;

            return reduce_segments<Accum, W>(
                queue, span.total_segments(), span.extent, geometry,
                Detail::AxisOpen<T, W, Transform>(in, span, transform),
                Detail::AxisStore<T, W, Finalize>(out, span.out_stride, span.extent, finalize),
                deps);
        }

    } // namespace Reduce
} // namespace SushiBLAS
