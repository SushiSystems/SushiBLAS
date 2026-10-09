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
#include <complex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/accumulator.hpp>

namespace SushiBLAS
{
    namespace Reduce
    {
        /**
         * @brief Holds the ceiling on the number of per-work-group partials.
         * @see modules/foundation/support/README.md
         */
        inline constexpr std::size_t K_MAX_GROUPS = 256;

        /**
         * @brief Holds the element count at which a CPU switches from one pass to two.
         * @see modules/foundation/support/README.md
         */
        inline constexpr std::size_t K_CPU_TWO_STAGE_MIN_ELEMENTS = 16384;

        /**
         * @brief Names the type a reduction over T accumulates in: FLOAT32 for HALF, T otherwise.
         * @see modules/foundation/support/README.md
         */
        template<typename T>
        using Accumulation = Kernels::AccumulatorOf<T>;

        /** @brief Names `Accumulation<T>::type`, the accumulation type itself. */
        template<typename T>
        using AccumulationOf = Kernels::Accumulator<T>;

        namespace Detail
        {
            /** @brief Returns whether x is a NaN; an integer never is. */
            template<typename R>
            [[nodiscard]] inline bool is_nan([[maybe_unused]] R x) noexcept
            {
                if constexpr (std::is_integral_v<R>)
                    return false;
                else
                    return sycl::isnan(x);
            }
        } // namespace Detail

        /**
         * @struct Plan
         * @brief Holds the launch geometry of one reduction.
         */
        struct Plan
        {
            /**
             * @brief Holds the work-groups in stage one, or 0 for the single-pass shape.
             * @see modules/foundation/support/README.md
             */
            std::size_t groups = 0;

            /** @brief Holds the work-items per work-group; always a power of two. */
            std::size_t local = 1;

            /** @brief Returns true when this plan is the single-pass sequential shape. */
            [[nodiscard]] bool sequential() const noexcept
            {
                return groups == 0;
            }
        };

        /**
         * @brief Returns the largest power of two not exceeding v, and never zero.
         * @see modules/foundation/support/README.md
         */
        [[nodiscard]] inline constexpr std::size_t floor_pow2(std::size_t v) noexcept
        {
            std::size_t p = 1;
            while (p * 2 <= v)
                p *= 2;
            return p;
        }

        /**
         * @brief Returns the work-group width stage one uses on @p p.
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
         * @see modules/foundation/support/README.md
         */
        [[nodiscard]] inline std::size_t planned_partials(const Device::Profile& p,
                                                          std::size_t n) noexcept
        {
            if (n == 0)
                return 0;
            if (p.is_cpu && n < K_CPU_TWO_STAGE_MIN_ELEMENTS)
                return 0;

            const std::size_t local = group_width(p);
            const std::size_t covering = (n + local - 1) / local;
            const std::size_t resident = p.compute_units * Device::K_WAVES_PER_COMPUTE_UNIT;

            const std::size_t groups = std::min({covering, resident, K_MAX_GROUPS});

            // One group has nothing to fold across, so it takes the single-pass shape.
            return groups > 1 ? groups : 0;
        }

        /**
         * @brief Returns the geometry to launch with, given the scratch actually available.
         * @param scratch_capacity Partials the caller's scratch buffer can hold.
         * @return The plan; Plan::sequential is true when there is no stage two.
         * @see modules/foundation/support/README.md
         */
        [[nodiscard]] inline Plan plan(const Device::Profile& p, std::size_t n,
                                       std::size_t scratch_capacity) noexcept
        {
            const std::size_t wanted = std::min(planned_partials(p, n), scratch_capacity);
            if (wanted < 2)
                return Plan{0, 1};
            return Plan{wanted, group_width(p)};
        }

        /**
         * @brief Holds a running sum that carries Neumaier's correction term.
         * @see modules/foundation/support/README.md
         */
        template<typename T>
        class CompensatedSum
        {
            public:
                /** @brief Returns the value an empty sum has. */
                [[nodiscard]] static constexpr T identity() noexcept
                {
                    return T(0);
                }

                /**
                 * @brief Combines two finished sums.
                 * @return `a + b`.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept
                {
                    return a + b;
                }

                /**
                 * @brief Folds one value into the running sum, tracking what was lost.
                 * @see modules/foundation/support/README.md
                 */
                void accumulate(T x) noexcept
                {
                    const T t = sum_ + x;

                    // The correction changes only while the total t is finite.
                    if (sycl::isfinite(t))
                    {
                        if (sycl::fabs(sum_) >= sycl::fabs(x))
                            comp_ += (sum_ - t) + x;
                        else
                            comp_ += (x - t) + sum_;
                    }

                    sum_ = t;
                }

                /**
                 * @brief Multiplies the running total and its correction by factor.
                 * @param factor Exact when it is a power of two.
                 * @see modules/foundation/support/README.md
                 */
                void rescale(T factor) noexcept
                {
                    sum_ *= factor;
                    comp_ *= factor;
                }

                /**
                 * @brief Returns the sum with its accumulated correction applied.
                 * @return `sum + compensation`, or the running total alone when it is not finite.
                 */
                [[nodiscard]] T value() const noexcept
                {
                    return sycl::isfinite(sum_) ? T(sum_ + comp_) : sum_;
                }

            private:
                /** @brief Holds the running total, as plain summation would compute it. */
                T sum_ = T(0);

                /** @brief Holds everything @ref sum_ has rounded away so far. */
                T comp_ = T(0);
        };

        /**
         * @brief Holds a running Euclidean norm that cannot overflow or underflow on the way.
         * @see modules/foundation/support/README.md
         */
        template<typename T>
        class ScaledSumOfSquares
        {
            public:
                /**
                 * @brief Returns the smallest value the scale may take: the smallest positive
                 *        normal value.
                 * @see modules/foundation/support/README.md
                 */
                [[nodiscard]] static T min_scale() noexcept
                {
                    return std::numeric_limits<T>::min();
                }

                /**
                 * @brief Returns the largest value the scale may take: the reciprocal of min_scale.
                 * @see modules/foundation/support/README.md
                 */
                [[nodiscard]] static T max_scale() noexcept
                {
                    return T(1) / std::numeric_limits<T>::min();
                }

                /** @brief Returns the value an empty norm has. */
                [[nodiscard]] static constexpr T identity() noexcept
                {
                    return T(0);
                }

                /**
                 * @brief Combines two finished partial norms with hypot.
                 * @see modules/foundation/support/README.md
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept
                {
                    return sycl::hypot(a, b);
                }

                /**
                 * @brief Folds one raw element or finished partial norm into the running norm.
                 * @see modules/foundation/support/README.md
                 */
                void accumulate(T x) noexcept
                {
                    const T a = sycl::fabs(x);

                    // An exact zero contributes nothing and skips climb_to.
                    if (a == T(0))
                        return;

                    // The scale climbs only when a leaves the current binade.
                    if (a >= hi_)
                        climb_to(a);

                    // climb_to's bounds keep a * inv_scale_ in range.
                    const T m = a * inv_scale_;
                    ssq_.accumulate(m * m);
                }

                /**
                 * @brief Returns the norm folded so far.
                 * @return `scale * sqrt(sum-of-relative-squares)`.
                 */
                [[nodiscard]] T value() const noexcept
                {
                    return scale_ * sycl::sqrt(ssq_.value());
                }

            private:
                /**
                 * @brief Raises the scale to the power of two at or below a, clamped to
                 *        [min_scale, max_scale], and rescales the total already folded to match.
                 * @pre a is above zero, at or above hi_, and not a NaN.
                 * @see modules/foundation/support/README.md
                 */
                void climb_to(T a) noexcept
                {
                    const T ceiling = max_scale();

                    // Both bounds are powers of two, so the clamped value keeps next in range.
                    T clamped = a;
                    if (clamped < min_scale())
                        clamped = min_scale();
                    if (clamped > ceiling)
                        clamped = ceiling;

                    const T next = sycl::ldexp(T(1), sycl::ilogb(clamped));
                    const T inv_next = T(1) / next;

                    // On the first value folded shrink is zero, which clears the empty sum.
                    const T shrink = scale_ * inv_next;
                    ssq_.rescale(shrink * shrink);

                    scale_ = next;
                    inv_scale_ = inv_next;

                    // hi_ is the next binade, 2 * next, and infinity at the ceiling.
                    hi_ = (next < ceiling) ? (next + next) : (next * T(4));
                }

                /** @brief Holds the power of two every folded value is expressed relative to. */
                T scale_ = T(0);

                /** @brief Holds `1 / scale_`, exact because @ref scale_ is a power of two. */
                T inv_scale_ = T(0);

                /** @brief Holds `2 * scale_`: the magnitude at which the scale must climb. */
                T hi_ = T(0);

                /** @brief Holds the compensated sum of `(x / scale_)^2` over everything folded. */
                CompensatedSum<T> ssq_{};
        };

        /**
         * @brief Holds a running product, folded in the same fixed order as CompensatedSum.
         * @see modules/foundation/support/README.md
         */
        template<typename T>
        class RunningProduct
        {
            public:
                /** @brief Returns the value an empty product has. */
                [[nodiscard]] static constexpr T identity() noexcept
                {
                    return T(1);
                }

                /**
                 * @brief Combines two finished products.
                 * @param a The left partial.
                 * @param b The right partial.
                 * @return `a * b`.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept
                {
                    return a * b;
                }

                /**
                 * @brief Folds one value into the running product.
                 * @param x The value to multiply in.
                 */
                void accumulate(T x) noexcept
                {
                    prod_ *= x;
                }

                /**
                 * @brief Returns the product so far.
                 * @return The running product.
                 */
                [[nodiscard]] T value() const noexcept
                {
                    return prod_;
                }

            private:
                /** @brief Holds the running product. */
                T prod_ = T(1);
        };

        /**
         * @brief Holds a running maximum that turns into NaN once any NaN is folded.
         * @see modules/foundation/support/README.md
         */
        template<typename T>
        class NaNPropagatingMax
        {
            public:
                /**
                 * @brief Returns the value an empty maximum has: -infinity, or lowest() when T
                 *        has no infinity.
                 */
                [[nodiscard]] static T identity() noexcept
                {
                    if constexpr (std::numeric_limits<T>::has_infinity)
                        return -std::numeric_limits<T>::infinity();
                    else
                        return std::numeric_limits<T>::lowest();
                }

                /**
                 * @brief Combines two finished maximums, in either order with the same result.
                 * @return NaN when either side is NaN, +0 over -0, otherwise the greater value.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept
                {
                    if (sycl::isnan(a))
                        return a;
                    if (sycl::isnan(b))
                        return b;
                    if (a == b)
                        return sycl::signbit(a) ? b : a;
                    return a < b ? b : a;
                }

                /** @brief Folds one value into the running maximum. */
                void accumulate(T x) noexcept
                {
                    best_ = combine(best_, x);
                }

                /** @brief Returns the maximum so far, NaN if a NaN was folded, or identity(). */
                [[nodiscard]] T value() const noexcept
                {
                    return best_;
                }

            private:
                /** @brief The greater value folded so far, or the first NaN met. */
                T best_ = identity();
        };

        /**
         * @brief Holds a running minimum that turns into NaN once any NaN is folded.
         * @see modules/foundation/support/README.md
         */
        template<typename T>
        class NaNPropagatingMin
        {
            public:
                /**
                 * @brief Returns the value an empty minimum has: +infinity, or max() when T has
                 *        no infinity.
                 */
                [[nodiscard]] static T identity() noexcept
                {
                    if constexpr (std::numeric_limits<T>::has_infinity)
                        return std::numeric_limits<T>::infinity();
                    else
                        return std::numeric_limits<T>::max();
                }

                /**
                 * @brief Combines two finished minimums, in either order with the same result.
                 * @return NaN when either side is NaN, -0 over +0, otherwise the smaller value.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept
                {
                    if (sycl::isnan(a))
                        return a;
                    if (sycl::isnan(b))
                        return b;
                    if (a == b)
                        return sycl::signbit(a) ? a : b;
                    return b < a ? b : a;
                }

                /** @brief Folds one value into the running minimum. */
                void accumulate(T x) noexcept
                {
                    best_ = combine(best_, x);
                }

                /** @brief Returns the minimum so far, NaN if a NaN was folded, or identity(). */
                [[nodiscard]] T value() const noexcept
                {
                    return best_;
                }

            private:
                /** @brief The smaller value folded so far, or the first NaN met. */
                T best_ = identity();
        };

        /**
         * @brief Holds a running sum over a type whose addition is exact.
         * @see modules/foundation/support/README.md
         */
        template<typename T>
        class ExactSum
        {
            public:
                /** @brief Returns the value an empty sum has. */
                [[nodiscard]] static constexpr T identity() noexcept
                {
                    return T(0);
                }

                /**
                 * @brief Combines two finished sums.
                 * @param a The left partial.
                 * @param b The right partial.
                 * @return `a + b`.
                 */
                [[nodiscard]] static constexpr T combine(T a, T b) noexcept
                {
                    return a + b;
                }

                /**
                 * @brief Folds one value into the running sum.
                 * @param x The value to add.
                 */
                void accumulate(T x) noexcept
                {
                    sum_ += x;
                }

                /**
                 * @brief Returns the sum so far.
                 * @return The running total.
                 */
                [[nodiscard]] T value() const noexcept
                {
                    return sum_;
                }

            private:
                /** @brief Holds the running total. */
                T sum_ = T(0);
        };

        /**
         * @brief Holds a complex running sum as two CompensatedSum, one per component.
         * @tparam C A std::complex type.
         * @see modules/foundation/support/README.md
         */
        template<typename C>
        class CompensatedComplexSum
        {
            public:
                /** @brief Returns the value an empty sum has. */
                [[nodiscard]] static C identity() noexcept
                {
                    return C(0);
                }

                /**
                 * @brief Combines two finished sums.
                 * @return `a + b`.
                 */
                [[nodiscard]] static C combine(C a, C b) noexcept
                {
                    return a + b;
                }

                /**
                 * @brief Folds one value into the running sum, component by component.
                 * @param x The value to add.
                 */
                void accumulate(C x) noexcept
                {
                    re_.accumulate(x.real());
                    im_.accumulate(x.imag());
                }

                /** @brief Returns the sum with both corrections applied. */
                [[nodiscard]] C value() const noexcept
                {
                    return C(re_.value(), im_.value());
                }

            private:
                /** @brief Holds the real component. */
                CompensatedSum<typename C::value_type> re_{};

                /** @brief Holds the imaginary component. */
                CompensatedSum<typename C::value_type> im_{};
        };

        /**
         * @struct IndexedValue
         * @brief Pairs a value with the index it was read from.
         */
        template<typename R>
        struct IndexedValue
        {
            /** @brief Holds the value compared. */
            R value;

            /** @brief Holds the element index, or -1 for the empty candidate. */
            std::int64_t index;
        };

        /**
         * @brief Holds the running (value, index) maximum under the iamax reference kernel's
         *        NaN rule.
         * @tparam P An IndexedValue type.
         * @see modules/foundation/support/README.md
         */
        template<typename P>
        class RunningArgMax
        {
            public:
                /** @brief Returns the empty candidate, which every real candidate beats. */
                [[nodiscard]] static P identity() noexcept
                {
                    return P{std::numeric_limits<decltype(P::value)>::lowest(), -1};
                }

                /**
                 * @brief Returns the winner of two candidates under a total order.
                 * @return A NaN at index 0 first, then the larger value, then the lower index;
                 *         a NaN elsewhere beats only the empty candidate.
                 * @see modules/foundation/support/README.md
                 */
                [[nodiscard]] static P combine(P a, P b) noexcept
                {
                    if (a.index < 0)
                        return b;
                    if (b.index < 0)
                        return a;

                    const bool a_nan = Detail::is_nan(a.value);
                    const bool b_nan = Detail::is_nan(b.value);

                    if (a_nan && a.index == 0)
                        return a;
                    if (b_nan && b.index == 0)
                        return b;
                    if (a_nan != b_nan)
                        return a_nan ? b : a;
                    if (!a_nan)
                    {
                        if (a.value > b.value)
                            return a;
                        if (b.value > a.value)
                            return b;
                    }
                    return a.index < b.index ? a : b;
                }

                /**
                 * @brief Folds one candidate into the running maximum.
                 * @param x The candidate to consider.
                 */
                void accumulate(P x) noexcept
                {
                    best_ = combine(best_, x);
                }

                /** @brief Returns the winning candidate, or identity() if nothing was folded. */
                [[nodiscard]] P value() const noexcept
                {
                    return best_;
                }

            private:
                /** @brief Holds the winning candidate so far. */
                P best_ = identity();
        };

        /**
         * @brief Holds the running (value, index) minimum under the NaN rule RunningArgMax uses.
         * @tparam P An IndexedValue type.
         * @see modules/foundation/support/README.md
         */
        template<typename P>
        class RunningArgMin
        {
            public:
                /** @brief Returns the empty candidate, which every real candidate beats. */
                [[nodiscard]] static P identity() noexcept
                {
                    return P{std::numeric_limits<decltype(P::value)>::max(), -1};
                }

                /**
                 * @brief Returns the winner of two candidates under a total order.
                 * @return A NaN at index 0 first, then the smaller value, then the lower index;
                 *         a NaN elsewhere beats only the empty candidate.
                 * @see modules/foundation/support/README.md
                 */
                [[nodiscard]] static P combine(P a, P b) noexcept
                {
                    if (a.index < 0)
                        return b;
                    if (b.index < 0)
                        return a;

                    const bool a_nan = Detail::is_nan(a.value);
                    const bool b_nan = Detail::is_nan(b.value);

                    if (a_nan && a.index == 0)
                        return a;
                    if (b_nan && b.index == 0)
                        return b;
                    if (a_nan != b_nan)
                        return a_nan ? b : a;
                    if (!a_nan)
                    {
                        if (a.value < b.value)
                            return a;
                        if (b.value < a.value)
                            return b;
                    }
                    return a.index < b.index ? a : b;
                }

                /** @brief Folds one candidate into the running minimum. */
                void accumulate(P x) noexcept
                {
                    best_ = combine(best_, x);
                }

                /** @brief Returns the winning candidate, or identity() if nothing was folded. */
                [[nodiscard]] P value() const noexcept
                {
                    return best_;
                }

            private:
                /** @brief Holds the winning candidate so far. */
                P best_ = identity();
        };

        /**
         * @brief Reduces indices `[0..n)` to `*out` as `finalize(fold(read(i)), n)` in a given geometry.
         * @pre geometry.sequential(), or partials holds geometry.groups values and geometry.local
         *      is a power of two the device accepts as a work-group size.
         * @param read Callable `(std::size_t i) -> W`, the input of element i.
         * @param finalize Callable `(W value, std::size_t n)`, applied once; converted to Out on store.
         * @see modules/foundation/support/README.md
         */
        template<template<typename> class Accum, typename W, typename Out,
                 typename Read, typename Finalize>
        [[nodiscard]] sycl::event reduce_indexed(sycl::queue& queue, const Plan& geometry,
                                                 std::size_t n, Read read, Out* out, W* partials,
                                                 Finalize finalize,
                                                 const std::vector<sycl::event>& deps)
        {
            if (geometry.sequential())
            {
                // The single-pass shape is one task walking the input in order.
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        Accum<W> acc;
                        for (std::size_t i = 0; i < n; ++i) acc.accumulate(read(i));
                        *out = static_cast<Out>(finalize(acc.value(), n));
                    });
                });
            }

            const std::size_t groups = geometry.groups;
            const std::size_t local = geometry.local;

            const sycl::event staged = queue.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);

                sycl::local_accessor<W, 1> tile(sycl::range<1>(local), h);

                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local),
                                                 sycl::range<1>(local)),
                               [=](sycl::nd_item<1> it)
                {
                    const std::size_t gid = it.get_global_id(0);
                    const std::size_t lid = it.get_local_id(0);
                    const std::size_t stride = groups * local;

                    // Work-item g owns indices g, g + stride, g + 2*stride.
                    Accum<W> acc;
                    for (std::size_t i = gid; i < n; i += stride)
                        acc.accumulate(read(i));

                    tile[lid] = acc.value();

                    sycl::group_barrier(it.get_group());

                    // A fixed-order halving tree; local is a power of two.
                    for (std::size_t off = local / 2; off > 0; off >>= 1)
                    {
                        if (lid < off)
                            tile[lid] = Accum<W>::combine(tile[lid], tile[lid + off]);
                        sycl::group_barrier(it.get_group());
                    }

                    if (lid == 0)
                        partials[it.get_group(0)] = tile[0];
                });
            });

            return queue.submit([&](sycl::handler& h)
            {
                h.depends_on(staged);
                h.single_task([=]()
                {
                    // Stage two folds the partials in ascending index order.
                    Accum<W> acc;
                    for (std::size_t g = 0; g < groups; ++g) acc.accumulate(partials[g]);
                    *out = static_cast<Out>(finalize(acc.value(), n));
                });
            });
        }

        /**
         * @brief Reduces indices `[0..n)` to `*out` in the geometry the queue's device plans.
         * @param partials Scratch for one W per work-group; may be null when partial_capacity is zero.
         * @return The event for the last submitted kernel.
         * @see modules/foundation/support/README.md
         */
        template<template<typename> class Accum, typename W, typename Out,
                 typename Read, typename Finalize>
        [[nodiscard]] sycl::event reduce_indexed(sycl::queue& queue, std::size_t n, Read read,
                                                 Out* out, W* partials,
                                                 std::size_t partial_capacity,
                                                 Finalize finalize,
                                                 const std::vector<sycl::event>& deps)
        {
            return reduce_indexed<Accum>(queue,
                                         plan(Device::profile_for(queue), n, partial_capacity),
                                         n, read, out, partials, finalize, deps);
        }

        /**
         * @brief Reduces `in[0..n)` to `*out` as `finalize(fold(transform(in[i])), n)` in a given geometry.
         * @pre The precondition of the geometry form of reduce_indexed.
         * @param transform Callable `(T) -> A`, applied to each element before the fold.
         * @param finalize Callable `(AccumulationOf<A> value, std::size_t n)`, applied once.
         * @see modules/foundation/support/README.md
         */
        template<template<typename> class Accum, typename T, typename A,
                 typename Transform, typename Finalize>
        [[nodiscard]] sycl::event reduce(sycl::queue& queue, const Plan& geometry, const T* in,
                                         A* out, std::size_t n, AccumulationOf<A>* partials,
                                         Transform transform, Finalize finalize,
                                         const std::vector<sycl::event>& deps)
        {
            using W = AccumulationOf<A>;

            return reduce_indexed<Accum>(queue, geometry, n,
                [=](std::size_t i) { return static_cast<W>(transform(in[i])); },
                out, partials, finalize, deps);
        }

        /**
         * @brief Reduces `in[0..n)` to `*out` in the geometry the queue's device plans.
         * @param partials Scratch for one AccumulationOf<A> per work-group; may be null when
         *        partial_capacity is zero.
         * @return The event for the last submitted kernel.
         * @see modules/foundation/support/README.md
         */
        template<template<typename> class Accum, typename T, typename A,
                 typename Transform, typename Finalize>
        [[nodiscard]] sycl::event reduce(sycl::queue& queue, const T* in, A* out,
                                         std::size_t n, AccumulationOf<A>* partials,
                                         std::size_t partial_capacity,
                                         Transform transform, Finalize finalize,
                                         const std::vector<sycl::event>& deps)
        {
            return reduce<Accum>(queue, plan(Device::profile_for(queue), n, partial_capacity),
                                 in, out, n, partials, transform, finalize, deps);
        }

    } // namespace Reduce
} // namespace SushiBLAS
