/****************************************************************/
/* reduce_axis.hpp                                              */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file reduce_axis.hpp
 * @brief The **segmented** reduction shape: many independent folds, one result
 * each, laid out by a stride pair.
 *
 * `reduce_1d.hpp` folds a whole tensor to one scalar and `launch_1d.hpp` maps
 * every element to its own output. This is the third shape, and the one neither
 * of those can express: N folds running side by side, each over its own slice of
 * the same buffer, producing a vector.
 *
 * **Why it exists.** A log-sum-exp shift has to be *per row*. Shifting a whole
 * batch by the batch's global maximum is exact in real arithmetic - log-softmax
 * is invariant to subtracting any constant from a row - and wrong in binary32,
 * because the invariance is destroyed by the exponential underflowing: a row
 * sitting 104 below the batch maximum has every `exp(x - global_max)` flush to
 * zero, its row sum is exactly zero, `log(0)` is `-inf`, and the row comes back
 * `+inf`. That is a silently wrong answer for gaps around 90-104 and a
 * catastrophically wrong one past it. The only fix is a maximum computed per
 * row, and a per-row maximum is this shape.
 *
 * **The layout is a stride pair, not an axis number.** @ref AxisSpan describes
 * "where do the segments start, and how do I walk one" in element counts, which
 * makes the axis, the `Core::Layout` and any non-unit stride a *caller's*
 * problem, resolved once on the host. Reducing along either axis of either
 * layout is the same kernel with different numbers, so there is no layout branch
 * in the inner loop and no second instantiation.
 *
 * **Two shapes, for the two regimes.** A segmented reduction has two independent
 * sources of parallelism and which one is available is a property of the input:
 *
 *   1. **Item per segment** (@ref AxisPlan::cooperative false). Each work-item
 *      owns one whole segment and folds it serially. The parallelism is the
 *      segment count, so this is what a tall, narrow reduction wants - a
 *      row-wise reduction over a batch, which is the case this file was written
 *      for. Goes through `Device::plan_1d`, so it inherits the CPU-flat /
 *      GPU-grid-stride split `launch_1d.hpp` measured.
 *   2. **Work-group per segment** (@ref AxisPlan::cooperative true). Each
 *      work-group owns one segment, grid-strides across it and folds through a
 *      halving tree over local memory. The parallelism is the segment *length*,
 *      which is what recovers the machine when there are fewer segments than it
 *      has execution units - a `[2, 10M]` row-wise sum would otherwise run on
 *      two work-items.
 *
 * Both are correct at every shape; the choice is only ever about speed. In
 * particular neither requires the segment length to relate to the work-group
 * width in any way - see @ref reduce_axis for what the tree does with a partial
 * group.
 *
 * **Determinism.** Same standard, and the same argument, as `reduce_1d.hpp`:
 * `sycl::reduction` and `sycl::reduce_over_group` are both unusable because the
 * specification fixes no combination order for them and floating-point addition
 * is not associative. Shape 1 has *no* cross-work-item combination at all - one
 * item folds one segment in ascending index order. Shape 2 combines through the
 * same hand-written fixed-offset halving tree stage one of `Reduce::reduce`
 * uses. In both, which elements a given lane folds and the order it folds them
 * in are pure functions of the launch geometry, and the geometry is a pure
 * function of the device profile and the span. Two runs of the same binary on
 * the same device therefore perform the same operations on the same values in
 * the same order. A second segment axis (@ref AxisSpan::segments2) leaves that
 * argument alone: it only changes which address a fold starts at, so the fold
 * order *within* a segment did not move and no combination across segments was
 * introduced.
 *
 * That is an argument from what the code does, not a measurement of a GPU - the
 * same honest limit `reduce_1d.hpp` records for itself. See
 * `docs/GPU_VALIDATION.md` §5.
 *
 * **No scratch buffer, and that is structural rather than lucky.** A whole-tensor
 * reduction needs one because its work-groups all contribute to a single answer
 * and something has to hold the partials between the two kernels. Here a
 * work-group's tree *is* its segment's answer, so there is nothing to combine
 * across groups, nothing to stage, and one kernel submission either way.
 */

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
         * @struct AxisSpan
         * @brief Where one segmented reduction's inputs and outputs live, in
         * element counts.
         *
         * Everything the kernel needs to address memory, and nothing about *why*:
         * an axis index, a `Core::Layout` and a strided view all collapse into
         * these five numbers on the host, so the device never sees any of them.
         * The element at position @p k of segment @p s is
         * `in[s * segment_stride + k * element_stride]`, and its result goes to
         * `out[s * out_stride]`.
         *
         * All strides are element counts and all are non-negative, which
         * `Tensor`'s stride-explicit constructor already guarantees for any view
         * this can be built from.
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
             * @brief How many *groups* of segments; 1 when there is only one axis
             * of segmentation.
             *
             * The second **segment** axis, not a second folded one - and which
             * of the two it is decides which loop changes. Folding two axes at
             * once needs nothing new: `[N*H*W, C]` is a zero-copy rank-2 view and
             * the existing five numbers fold axis 0 to `[C]`. What the five
             * numbers cannot express is a segment identified by a *pair*:
             * `[N,H,W,C] -> [N,C]` has one result per `(n, c)`, whose input
             * offset is `n * H*W*C + c`, and no single `segment_stride` produces
             * that from a single index.
             *
             * Segment `(s2, s)` starts at `in[s2 * segment_stride2 + s * segment_stride]`
             * and its result goes to `out[(s2 * segments + s) * out_stride]`.
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
         * @brief Picks between the two segmented shapes for @p segments folds of
         * @p extent elements on @p p.
         *
         * The rule is a parallelism argument and nothing else, because both
         * shapes are correct at every size:
         *
         * - **Enough segments to fill the machine?** Then one item per segment,
         *   which is strictly cheaper - no local memory, no barriers, one
         *   coalesced grid-stride launch, and the fold order is trivially fixed.
         *   "Enough" is one work-item per hardware thread on a CPU, and
         *   `K_WAVES_PER_COMPUTE_UNIT` resident groups' worth on a GPU, matching
         *   what `Device::plan_1d` targets for a flat launch.
         * - **Otherwise, is a segment long enough to be worth a work-group?** A
         *   group whose segment is shorter than the group is mostly idle lanes
         *   paying `log2(local)` pairs of barriers, so the tree only earns its
         *   keep once every lane has at least one element to fold. One full
         *   group-width pass is that threshold.
         * - **Otherwise** there is nothing to recover; take the cheap shape.
         *
         * @warning The GPU side of this is **derived, not measured** - this
         *          project has no GPU (see `docs/GPU_VALIDATION.md`). The
         *          threshold is a floor below which the cooperative shape is
         *          certainly wrong, not a tuned crossover, and it is the first
         *          knob to reach for if a row-wise reduction underperforms on
         *          real hardware. It cannot make an answer wrong: the shape is a
         *          pure function of the device and the span, so it is the same on
         *          every run, and both shapes compute the same thing.
         *
         * @param p The target device's profile.
         * @param segments How many independent folds.
         * @param extent How many elements each fold covers.
         * @return The geometry; @ref AxisPlan::cooperative says which shape.
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
         * @brief Reduces each segment of @p span to one element of @p out,
         * deterministically.
         *
         * Computes `out[s] = finalize(fold(transform(in[s][k])), extent)` for
         * every segment @p s, where `fold` is @p Accum's operation applied in an
         * order fixed entirely by the launch geometry. One kernel submission,
         * whichever shape @ref plan_axis chose.
         *
         * A segment shorter than the work-group is still correct on the
         * cooperative shape: a lane with no element of its own contributes
         * `Accum<T>{}.value()`, which is the accumulator's identity by
         * construction, and combining the identity changes nothing. @ref plan_axis
         * declines that shape below one full pass for cost, not correctness.
         *
         * @tparam Accum The accumulation policy - @ref CompensatedSum,
         *         @ref RunningMax, @ref RunningProduct - a class template on the
         *         element type supplying `identity()`, `combine(a, b)`,
         *         `accumulate(x)` and `value()`.
         * @tparam T The element type, deduced.
         * @tparam Transform The per-element transform type, deduced.
         * @tparam Finalize The finalizer type, deduced.
         * @param queue The queue to submit to.
         * @param in The input array; fold `s` starts at
         *        `in + (s / segments) * segment_stride2 + (s % segments) * segment_stride`,
         *        which is `in + s * segment_stride` whenever there is no second
         *        segment axis.
         * @param out Where the per-segment results go.
         * @param span The segment layout, in element counts.
         * @param geometry Which shape to launch, normally @ref plan_axis's answer.
         *        Taken rather than decided here so that the planner and the launcher
         *        stay separable: a test can then drive both shapes over one input and
         *        check they agree, which is the only way the cooperative tree gets
         *        exercised at a size the planner would not send to it.
         * @param transform Callable `(T) -> T`, applied to each element before it
         *        is folded in. Pass the identity when there is nothing to do.
         * @param finalize Callable `(T value, std::size_t n) -> T`, applied once
         *        per segment, with @p n the segment's own element count - so a
         *        per-axis mean is this seam plus `v / n` rather than a second
         *        kernel.
         * @param deps Events this submission must depend on.
         * @return The event for the submitted kernel.
         *
         * @warning @p transform and @p finalize are inlined into a SYCL device
         *          kernel: they must be device-copyable, capture by value, hold no
         *          host pointers, allocate nothing and throw nothing.
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
                // One item per segment, over the same flat/grid-stride launch
                // shape every elementwise op takes - so a CPU covers the segments
                // in one pass of a wide group and a GPU walks a capped, resident
                // grid. The stride loop's `s < total` test is the bounds check.
                const Device::Grid1D grid = Device::plan_1d(Device::profile_for(queue), total);
                const std::size_t stride = grid.stride();

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t s = it.get_global_id(0); s < total; s += stride)
                        {
                            // Ascending index order within the segment, and no
                            // other work-item touches it. There is nothing here
                            // whose order the SYCL specification leaves free.
                            Accum<T> acc;
                            // One integer division per fold, against an inner
                            // loop of `extent` elements. Not worth a specialised
                            // segments2 == 1 kernel, which would be a second
                            // instantiation of every reduction in the library.
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

                    // One integer division per fold, against an inner
                    // loop of `extent` elements. Not worth a specialised
                    // segments2 == 1 kernel, which would be a second
                    // instantiation of every reduction in the library.
                    const std::size_t s2 = s / segments;
                    const std::size_t s1 = s % segments;
                    const T* base = in + s2 * segment_stride2 + s1 * segment_stride;

                    // A stride-`local` slice of one segment: lane l owns
                    // positions l, l + local, l + 2*local, ... Which positions
                    // those are comes only from the geometry, so it is the same
                    // on every run, and consecutive lanes read consecutive
                    // addresses at every step.
                    Accum<T> acc;
                    for (std::size_t k = lid; k < extent; k += local)
                        acc.accumulate(transform(base[k * element_stride]));

                    tile[lid] = acc.value();

                    sycl::group_barrier(it.get_group());

                    // The same hand-written halving tree as stage one of
                    // Reduce::reduce, and NOT sycl::reduce_over_group, for the
                    // reason recorded there: the spec lets that function combine
                    // in any order it likes, which for a non-associative operator
                    // means different bits on different launches of one kernel.
                    // `local` is a power of two (see Reduce::group_width), which
                    // is what makes the halving visit every lane exactly once.
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
