/**************************************************************************/
/* reduce_1d.hpp                                                          */
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
 * @file reduce_1d.hpp
 * @brief Declares the deterministic whole-tensor reduction and the accumulators it folds with.
 * @author Mustafa Garip
 */

#pragma once

#include <limits>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/device_policy.hpp>

namespace SushiBLAS
{
    namespace Reduce
    {
        /**
         * @brief Holds the ceiling on the number of per-work-group partials.
         * @see include/SushiBLAS/engine/README.md
         */
        inline constexpr std::size_t K_MAX_GROUPS = 256;

        /**
         * @brief Holds the element count at which a CPU switches from one pass to two.
         * @see include/SushiBLAS/engine/README.md
         */
        inline constexpr std::size_t K_CPU_TWO_STAGE_MIN_ELEMENTS = 16384;

        /**
         * @struct Plan
         * @brief The launch geometry of one reduction.
         */
        struct Plan
        {
            /**
             * @brief Work-groups in stage one, or 0 for the single-pass shape.
             *
             * Zero is not "no work": it selects a wholly different kernel, one
             * sequential pass with no partials, no local memory and no stage two.
             */
            std::size_t groups = 0;

            /** @brief Work-items per work-group; always a power of two. */
            std::size_t local = 1;

            /** @brief True when this plan is the single-pass sequential shape. */
            [[nodiscard]] bool sequential() const noexcept { return groups == 0; }
        };

        /**
         * @brief Returns the largest power of two not exceeding v, and never zero.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline constexpr std::size_t floor_pow2(std::size_t v) noexcept
        {
            std::size_t p = 1;
            while (p * 2 <= v) p *= 2;
            return p;
        }

        /**
         * @brief The work-group width stage one uses on @p p.
         *
         * @param p The target device's profile.
         * @return A power-of-two work-group size the device will accept.
         */
        [[nodiscard]] inline std::size_t group_width(const Device::Profile& p) noexcept
        {
            return floor_pow2(std::min(Device::work_group_size(p), p.max_work_group_size));
        }

        /**
         * @brief Returns how many partials a two-stage reduction over n elements produces on p.
         * @return The partial count, or 0 to request the single-pass shape.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline std::size_t planned_partials(const Device::Profile& p,
                                                          std::size_t n) noexcept
        {
            if (n == 0) return 0;
            if (p.is_cpu && n < K_CPU_TWO_STAGE_MIN_ELEMENTS) return 0;

            const std::size_t local = group_width(p);
            const std::size_t covering = (n + local - 1) / local;
            const std::size_t resident = p.compute_units * Device::K_WAVES_PER_COMPUTE_UNIT;

            const std::size_t groups = std::min({covering, resident, K_MAX_GROUPS});

            // One group has nothing to fold across, so it takes the single-pass shape; see include/SushiBLAS/engine/README.md.
            return groups > 1 ? groups : 0;
        }

        /**
         * @brief Returns the geometry to launch with, given the scratch actually available.
         * @param scratch_capacity Partials the caller's scratch buffer can hold.
         * @return The plan; Plan::sequential is true when there is no stage two.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline Plan plan(const Device::Profile& p, std::size_t n,
                                       std::size_t scratch_capacity) noexcept
        {
            const std::size_t wanted = std::min(planned_partials(p, n), scratch_capacity);
            if (wanted < 2) return Plan{0, 1};
            return Plan{wanted, group_width(p)};
        }

        /**
         * @brief Holds a running sum that carries Neumaier's correction term.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T>
        class CompensatedSum
        {
            public:
                /** @brief The value an empty sum has. */
                [[nodiscard]] static constexpr T identity() noexcept { return T(0); }

                /**
                 * @brief Combines two finished sums.
                 * @param a The left partial.
                 * @param b The right partial.
                 * @return `a + b`.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept { return a + b; }

                /**
                 * @brief Folds one value into the running sum, tracking what was lost.
                 * @param x The value to add.
                 */
                void accumulate(T x) noexcept
                {
                    const T t = sum_ + x;

                    // The larger operand keeps its bits, so the correction recovers what the smaller one lost; see include/SushiBLAS/engine/README.md.
                    if (sycl::fabs(sum_) >= sycl::fabs(x)) comp_ += (sum_ - t) + x;
                    else                                   comp_ += (x - t) + sum_;

                    sum_ = t;
                }

                /**
                 * @brief Multiplies the running total and its correction by factor.
                 * @param factor Exact when it is a power of two.
                 * @see include/SushiBLAS/engine/README.md
                 */
                void rescale(T factor) noexcept
                {
                    sum_ *= factor;
                    comp_ *= factor;
                }

                /**
                 * @brief The sum with its accumulated correction applied.
                 * @return `sum + compensation`.
                 */
                [[nodiscard]] T value() const noexcept { return sum_ + comp_; }

            private:
                /** @brief The running total, as plain summation would compute it. */
                T sum_ = T(0);

                /** @brief Everything @ref sum_ has rounded away so far. */
                T comp_ = T(0);
        };

        /**
         * @brief Holds a running Euclidean norm that cannot overflow or underflow on the way.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T>
        class ScaledSumOfSquares
        {
            public:
                /**
                 * @brief Returns the smallest value the scale may take: the smallest positive normal value.
                 * @see include/SushiBLAS/engine/README.md
                 */
                [[nodiscard]] static T min_scale() noexcept { return std::numeric_limits<T>::min(); }

                /**
                 * @brief Returns the largest value the scale may take: the reciprocal of min_scale.
                 * @see include/SushiBLAS/engine/README.md
                 */
                [[nodiscard]] static T max_scale() noexcept { return T(1) / std::numeric_limits<T>::min(); }

                /** @brief The value an empty norm has. */
                [[nodiscard]] static constexpr T identity() noexcept { return T(0); }

                /**
                 * @brief Combines two finished partial norms with hypot.
                 * @see include/SushiBLAS/engine/README.md
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept { return sycl::hypot(a, b); }

                /**
                 * @brief Folds one raw element or finished partial norm into the running norm.
                 * @see include/SushiBLAS/engine/README.md
                 */
                void accumulate(T x) noexcept
                {
                    const T a = sycl::fabs(x);

                    // An exact zero contributes nothing and skips climb_to; see include/SushiBLAS/engine/README.md.
                    if (a == T(0)) return;

                    // The scale climbs only when a leaves the current binade; see include/SushiBLAS/engine/README.md.
                    if (a >= hi_) climb_to(a);

                    // climb_to's bounds keep a * inv_scale_ in range; see include/SushiBLAS/engine/README.md.
                    const T m = a * inv_scale_;
                    ssq_.accumulate(m * m);
                }

                /**
                 * @brief The norm folded so far.
                 * @return `scale * sqrt(sum-of-relative-squares)`.
                 */
                [[nodiscard]] T value() const noexcept { return scale_ * sycl::sqrt(ssq_.value()); }

            private:
                /**
                 * @brief Raises the scale to a and rescales the total already folded to match.
                 * @pre a is above zero, at or above hi_, and not a NaN.
                 * @see include/SushiBLAS/engine/README.md
                 */
                void climb_to(T a) noexcept
                {
                    const T ceiling = max_scale();

                    // The scale is clamped so that its reciprocal is normal; see include/SushiBLAS/engine/README.md.
                    T next = a;
                    if (next < min_scale()) next = min_scale();
                    if (next > ceiling) next = ceiling;

                    const T inv_next = T(1) / next;

                    // On the first value folded shrink is zero, which clears the empty sum; see include/SushiBLAS/engine/README.md.
                    const T shrink = scale_ * inv_next;
                    ssq_.rescale(shrink * shrink);

                    scale_ = next;
                    inv_scale_ = inv_next;

                    // hi_ is 2 * next, and 4 * next (infinity) at the ceiling; see include/SushiBLAS/engine/README.md.
                    hi_ = (next < ceiling) ? (next + next) : (next * T(4));
                }

                /** @brief The power of two every folded value is expressed relative to. */
                T scale_ = T(0);

                /** @brief `1 / scale_`, exact because @ref scale_ is a clamped power of two. */
                T inv_scale_ = T(0);

                /** @brief `2 * scale_`: the magnitude at which the scale must climb. */
                T hi_ = T(0);

                /** @brief Compensated sum of `(x / scale_)^2` over everything folded. */
                CompensatedSum<T> ssq_{};
        };

        /**
         * @brief Holds a running product, folded in the same fixed order as CompensatedSum.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T>
        class RunningProduct
        {
            public:
                /** @brief The value an empty product has. */
                [[nodiscard]] static constexpr T identity() noexcept { return T(1); }

                /**
                 * @brief Combines two finished products.
                 * @param a The left partial.
                 * @param b The right partial.
                 * @return `a * b`.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept { return a * b; }

                /**
                 * @brief Folds one value into the running product.
                 * @param x The value to multiply in.
                 */
                void accumulate(T x) noexcept { prod_ *= x; }

                /**
                 * @brief The product so far.
                 * @return The running product.
                 */
                [[nodiscard]] T value() const noexcept { return prod_; }

            private:
                /** @brief The running product. */
                T prod_ = T(1);
        };

        /**
         * @brief Holds a running maximum in the shape the folding seams accumulate in.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T>
        class RunningMax
        {
            public:
                /**
                 * @brief Returns the lowest finite value of T, the value an empty maximum has.
                 * @see include/SushiBLAS/engine/README.md
                 */
                [[nodiscard]] static T identity() noexcept
                {
                    return std::numeric_limits<T>::lowest();
                }

                /**
                 * @brief Combines two finished maxima.
                 * @param a The left partial.
                 * @param b The right partial.
                 * @return The greater, keeping @p a on a tie or a NaN.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept { return a < b ? b : a; }

                /**
                 * @brief Folds one value into the running maximum.
                 * @param x The value to consider.
                 */
                void accumulate(T x) noexcept { if (max_ < x) max_ = x; }

                /**
                 * @brief The maximum so far.
                 * @return The running maximum, or @ref identity if nothing was folded.
                 */
                [[nodiscard]] T value() const noexcept { return max_; }

            private:
                /** @brief The largest value folded so far. */
                T max_ = std::numeric_limits<T>::lowest();
        };

        /**
         * @brief Holds a running sum over a type whose addition is exact.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T>
        class ExactSum
        {
            public:
                /** @brief The value an empty sum has. */
                [[nodiscard]] static constexpr T identity() noexcept { return T(0); }

                /**
                 * @brief Combines two finished sums.
                 * @param a The left partial.
                 * @param b The right partial.
                 * @return `a + b`.
                 */
                [[nodiscard]] static constexpr T combine(T a, T b) noexcept { return a + b; }

                /**
                 * @brief Folds one value into the running sum.
                 * @param x The value to add.
                 */
                void accumulate(T x) noexcept { sum_ += x; }

                /**
                 * @brief The sum so far.
                 * @return The running total.
                 */
                [[nodiscard]] T value() const noexcept { return sum_; }

            private:
                /** @brief The running total. */
                T sum_ = T(0);
        };

        /**
         * @brief Reduces `in[0..n)` to `*out` as `finalize(fold(transform(in[i])), n)`, in a fixed order.
         * @param partials Scratch for one value per work-group; may be null when partial_capacity is zero.
         * @param transform Callable `(T) -> A`, applied to each element before the fold.
         * @param finalize Callable `(A value, std::size_t n) -> A`, applied once to the folded result.
         * @return The event for the last submitted kernel.
         * @see include/SushiBLAS/engine/README.md
         */
        template<template<typename> class Accum, typename T, typename A,
                 typename Transform, typename Finalize>
        [[nodiscard]] sycl::event reduce(sycl::queue& queue, const T* in, A* out,
                                         std::size_t n, A* partials,
                                         std::size_t partial_capacity,
                                         Transform transform, Finalize finalize,
                                         const std::vector<sycl::event>& deps)
        {
            const Plan geometry = plan(Device::profile_for(queue), n, partial_capacity);

            if (geometry.sequential())
            {
                // The single-pass shape is one task walking the input in order; see include/SushiBLAS/engine/README.md.
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        Accum<A> acc;
                        for (std::size_t i = 0; i < n; ++i) acc.accumulate(transform(in[i]));
                        *out = finalize(acc.value(), n);
                    });
                });
            }

            const std::size_t groups = geometry.groups;
            const std::size_t local = geometry.local;

            const sycl::event staged = queue.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);

                sycl::local_accessor<A, 1> tile(sycl::range<1>(local), h);

                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local),
                                                 sycl::range<1>(local)),
                               [=](sycl::nd_item<1> it)
                {
                    const std::size_t gid = it.get_global_id(0);
                    const std::size_t lid = it.get_local_id(0);
                    const std::size_t stride = groups * local;

                    // Work-item g owns indices g, g + stride, g + 2*stride; see include/SushiBLAS/engine/README.md.
                    Accum<A> acc;
                    for (std::size_t i = gid; i < n; i += stride)
                        acc.accumulate(transform(in[i]));

                    tile[lid] = acc.value();

                    sycl::group_barrier(it.get_group());

                    // A fixed-order halving tree; local is a power of two; see include/SushiBLAS/engine/README.md.
                    for (std::size_t off = local / 2; off > 0; off >>= 1)
                    {
                        if (lid < off) tile[lid] = Accum<A>::combine(tile[lid], tile[lid + off]);
                        sycl::group_barrier(it.get_group());
                    }

                    if (lid == 0) partials[it.get_group(0)] = tile[0];
                });
            });

            return queue.submit([&](sycl::handler& h)
            {
                h.depends_on(staged);
                h.single_task([=]()
                {
                    // Stage two folds the partials in ascending index order; see include/SushiBLAS/engine/README.md.
                    Accum<A> acc;
                    for (std::size_t g = 0; g < groups; ++g) acc.accumulate(partials[g]);
                    *out = finalize(acc.value(), n);
                });
            });
        }

    } // namespace Reduce
} // namespace SushiBLAS
