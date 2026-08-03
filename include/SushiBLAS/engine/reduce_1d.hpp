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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file reduce_1d.hpp
 * @brief The whole-tensor reduction shape, for the reductions whose value
 * depends on the order they combine in.
 *
 * `launch_1d.hpp` does this job for the per-element families; this is its
 * counterpart for the folding ones, and it exists for a correctness reason
 * rather than a performance one.
 *
 * **Why not `sycl::reduction`.** Every additive reduction in this library used
 * to be a one-line `sycl::reduction(p, T(0), sycl::plus<T>())`. The SYCL
 * specification does not fix the order in which such a reduction combines its
 * partial results, and floating-point addition is not associative, so the order
 * is visible in the result's bits and the implementation is free to change it
 * with the launch configuration, the driver version, or the device. SushiBLAS
 * links `sushiruntime_deterministic_fp` PUBLIC, whose guarantee is that the same
 * binary on the same device produces byte-identical results - so that guarantee
 * was resting on an implementation detail nobody had promised. It happened to
 * hold on this project's OpenCL CPU device. On the NVIDIA GPUs this project
 * targets, where a reduction's combination order genuinely does vary with the
 * grid, it would have broken exactly where the project was going.
 *
 * **What replaces it.** A two-stage reduction whose every step is fixed by the
 * geometry:
 *
 *   1. Each work-item walks a grid-strided slice of the input and accumulates it
 *      serially, with @ref CompensatedSum's Neumaier correction. The slice, and
 *      the order within it, are pure functions of the launch shape.
 *   2. One work-group's items are folded into one partial by a hand-written
 *      halving tree over local memory - **not** `sycl::reduce_over_group`, whose
 *      order the spec leaves just as free as `sycl::reduction`'s.
 *   3. A single task sums the fixed number of partials in ascending index order.
 *
 * Two runs of the same binary on the same device therefore perform the same
 * additions on the same values in the same order, which is what byte equality
 * actually requires.
 *
 * **On a CPU, below a measured size threshold, there is no stage two at all** -
 * one sequential pass is both faster and has no cross-work-item combination to
 * pin down. Above it, the two-stage shape wins by more than an order of
 * magnitude. See @ref planned_partials for the measurements, and @ref reduce for
 * the one caveat about what "sequential" buys.
 */

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
         * @brief Hard ceiling on the number of per-work-group partials.
         *
         * Stage two walks the partials sequentially in one task, so the count is a
         * direct latency cost there; and the scratch buffer that holds them is sized
         * from it at record time. 256 partials is enough to fill any GPU this project
         * will meet (a 108-SM part at eight resident groups per SM wants 864 groups
         * for a *saturating* launch, but a reduction is bandwidth-bound and 256 groups
         * of 256 items is already 65,536 work-items in flight), and 256 sequential
         * adds is a rounding error against reading the input at all.
         */
        inline constexpr std::size_t K_MAX_GROUPS = 256;

        /**
         * @brief Element count at which a CPU switches from one pass to two.
         *
         * Both shapes are deterministic, so this threshold is a pure performance
         * choice - but it is a *measured* one, and it is not the answer that was
         * expected. See @ref planned_partials for the numbers.
         *
         * It does not weaken the reproducibility guarantee. The shape is a function of
         * the device and the element count only, both fixed for a given call, so two
         * runs of the same reduction always take the same branch and perform the same
         * additions in the same order. A *different* size taking a different shape is
         * not a determinism problem, because it is a different computation.
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
         * @brief The largest power of two not exceeding @p v, and never zero.
         *
         * The halving tree in stage one folds `local / 2`, `local / 4`, ... down to
         * one, which only visits every element when `local` is a power of two. A
         * non-power-of-two group width would silently drop the odd items - a wrong
         * answer, not a slow one - so the width is rounded here rather than assumed.
         *
         * @param v The upper bound.
         * @return A power of two in [1, @p v].
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
         * @brief How many partials a two-stage reduction over @p n elements would
         * produce on @p p - and zero when the two-stage shape is the wrong one.
         *
         * This is the function `Dispatch::execute_reduction` sizes its scratch buffer
         * from, at task-record time, and it is also what selects between the library's
         * two reduction shapes. It depends only on the device and the element count,
         * both of which are fixed for a given call, so the choice is reproducible.
         *
         * **On a CPU it depends on the size, and the crossover was measured, not
         * guessed.** The expectation going in was that a CPU should always take the
         * single pass: one work-item, one order, nothing to argue about, and none of
         * the cost of a work-group tree, a local allocation, two barriers per level and
         * a second kernel submission. That turned out to be true only for small inputs.
         * Measured on this project's OpenCL CPU device (12 compute units), best of
         * seven runs of 200 recorded-and-executed `sum` launches, single-pass against
         * two-stage:
         *
         * |        n | single pass | two stage |
         * |---------:|------------:|----------:|
         * |    1,024 |      7.2 ms |   10.7 ms |
         * |    4,096 |      8.3 ms |   10.3 ms |
         * |   16,384 |     10.1 ms |   10.4 ms |
         * |   32,768 |     13.4 ms |   10.4 ms |
         * |   65,536 |     19.7 ms |    9.9 ms |
         * |  262,144 |     53.6 ms |   11.8 ms |
         * | 1,048,576|    195.7 ms |   16.6 ms |
         *
         * Below the crossover the single pass wins by up to 1.5x, because at that size
         * the whole reduction is launch overhead and the two-stage shape pays it twice.
         * Above it the single pass is doing what it says - one thread, one element at a
         * time - while eleven other cores sit idle, and by 1M elements that costs 11.8x.
         * The two shapes are level at 16K, which is where
         * @ref K_CPU_TWO_STAGE_MIN_ELEMENTS sits.
         *
         * A GPU always takes the two-stage shape: there a single task uses one lane of
         * one SM and leaves the entire machine idle at every size.
         *
         * @param p The target device's profile.
         * @param n How many elements are being reduced.
         * @return The partial count, or 0 to request the single-pass shape.
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

            // One group is a tree with nothing to fold across, so it is the single-pass
            // shape wearing a more expensive hat.
            return groups > 1 ? groups : 0;
        }

        /**
         * @brief The geometry to launch with, given the scratch actually available.
         *
         * @ref planned_partials is asked at record time and this is asked at execution
         * time, so in principle they could see different devices - the scratch is sized
         * from the device the *data* lives on, and the task is run on the device the
         * scheduler routes it to. Clamping to @p scratch_capacity here means such a
         * disagreement costs performance and never correctness: too little scratch
         * falls back to the single pass rather than writing past the buffer.
         *
         * @param p The device the kernel is actually being submitted to.
         * @param n How many elements are being reduced.
         * @param scratch_capacity Partials the caller's scratch buffer can hold.
         * @return The plan; @ref Plan::sequential is true when there is no stage two.
         */
        [[nodiscard]] inline Plan plan(const Device::Profile& p, std::size_t n,
                                       std::size_t scratch_capacity) noexcept
        {
            const std::size_t wanted = std::min(planned_partials(p, n), scratch_capacity);
            if (wanted < 2) return Plan{0, 1};
            return Plan{wanted, group_width(p)};
        }

        /**
         * @class CompensatedSum
         * @brief A running sum carrying Neumaier's correction term.
         *
         * Plain summation throws away the low-order bits of every addition whose
         * operands differ in magnitude, and over a long input those losses accumulate
         * into a result that can be arbitrarily wrong: summing 0.1f a million times
         * naively lands near 100958 instead of 100000. This keeps a second register
         * holding exactly what each addition lost and adds it back at the end.
         *
         * **Neumaier, not Kahan.** Kahan's correction assumes the running total is the
         * larger operand, which is true for a well-behaved input and false the moment a
         * single large value arrives before many small ones - precisely the case
         * `1e8 + 1 + 1 + ...` that a gradient norm or a loss accumulation looks like.
         * Neumaier tests which operand is larger and is exact in both directions, for
         * one extra comparison per element in a kernel that is waiting on memory
         * anyway.
         *
         * @tparam T The element type. The correction is only as good as T's own
         *         precision, so it buys the most for `float` and comparatively little
         *         for `sycl::half`, whose ten mantissa bits leave little to recover.
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

                    // Whichever operand is larger keeps its bits; the smaller one is the
                    // one that got rounded off, and `(larger - t) + smaller` recovers
                    // exactly the part that did not fit.
                    if (sycl::fabs(sum_) >= sycl::fabs(x)) comp_ += (sum_ - t) + x;
                    else                                   comp_ += (x - t) + sum_;

                    sum_ = t;
                }

                /**
                 * @brief Multiplies the running total, correction and all, by @p factor.
                 *
                 * For an accumulator whose terms are expressed relative to a scale that can
                 * change mid-fold - @ref ScaledSumOfSquares is the one in this library - the
                 * total already folded has to move with it. Doing that here rather than in
                 * the caller is what keeps the correction term correct: `comp_` is in the
                 * same units as `sum_`, so rescaling one and not the other would quietly
                 * stop compensating anything.
                 *
                 * @param factor The multiplier to apply. Exact when it is a power of two,
                 *        which is the only way @ref ScaledSumOfSquares ever calls this.
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
         * @class ScaledSumOfSquares
         * @brief A running Euclidean norm that cannot overflow or underflow on the way.
         *
         * The accumulation policy behind `nrm2`, and the reason it is a policy rather
         * than a `transform` is that squaring cannot be made safe one element at a time.
         * `sqrt(fold(x * x))` is the obvious spelling and it is wrong at both ends of the
         * range: an fp32 element past ~1.8e19 squares to infinity, so the norm of a vector
         * whose largest entry is 1e20 comes back `inf` when the true answer is about 1e20;
         * and an element below ~1e-22 squares to zero, so a vector of denormals comes back
         * `0` when the true answer is a perfectly representable denormal.
         *
         * Neither is academic in a training loop. Gradient-norm clipping calls `nrm2`, and
         * a gradient that has already exploded is precisely the input that overflows - so
         * the mechanism meant to rescue the run is the one that returns `inf` and ends it.
         *
         * **What it does instead**, which is what reference BLAS's `snrm2`/`dnrm2` do: it
         * keeps a running `scale_` and accumulates `sum((x / scale_)^2)`, so what is
         * squared is always a value near 1 however large or small the input is. The
         * finalised norm is `scale_ * sqrt(sum)`, and that product only overflows when the
         * true answer does.
         *
         * **The scale climbs by binade, and is applied by multiplying.** Two details, both
         * of which are about cost rather than about the mathematics. The scale is raised
         * only when an element reaches *twice* it, not merely when it beats the running
         * maximum, which caps the number of rescales at one per binade - about 250 for fp32
         * however long the vector is - instead of one per element for a sorted input. And
         * the reciprocal is taken once per rescale so that the per-element step is a
         * multiply, not the divide the textbook form performs. Measured over 100 launches
         * of a 1M-element fp32 `nrm2` on this project's OpenCL CPU device: 5.9 ms for the
         * naive form this replaced, 17.9 ms for the textbook divide-per-element one, 11.0 ms
         * for this. What is left over the naive form is the per-element `fabs`, compare and
         * multiply; see @ref climb_to for the two more elaborate variants that were tried
         * and measured *slower*, and for why the remaining inexactness does not accumulate.
         *
         * **The scale is bounded**, to `[min(), 1 / min()]` - for fp32 from 2^-126 to
         * 2^126. That is what keeps its reciprocal a normal number, and it costs nothing: a
         * magnitude below the bottom of that window still scales to a value whose square is
         * comfortably normal, and one above the top still scales to at most 4. It is also
         * what lets the hot path drop an `isfinite` test. An infinity clamps to the
         * ceiling, then scales to infinity and saturates the sum, which is the answer BLAS
         * gives; a NaN fails the `>=` that would rescale, then propagates through the
         * multiply. Neither needs a branch of its own.
         *
         * **How it fits @ref reduce.** `value()` returns a finished partial *norm* - a
         * single `T`, not the (scale, sum) pair - and @ref combine merges two of them with
         * `hypot`, which is itself overflow-safe. So the two-stage reduction needs no
         * wider partials buffer and no change to @ref reduce at all: stage one folds
         * elements, the work-group tree merges partial norms, and stage two folds those
         * partial norms back through @ref accumulate, which is correct because the norm of
         * a set of partial norms is the norm of the whole.
         *
         * **Determinism** is unchanged. Every branch here is a function of the values
         * folded and the order they arrive in, and @ref reduce fixes that order from the
         * geometry alone.
         *
         * @tparam T The element type.
         */
        template<typename T>
        class ScaledSumOfSquares
        {
            public:
                /**
                 * @brief Smallest value the scale is allowed to take.
                 *
                 * The smallest positive *normal* value, which for every IEEE format is
                 * exactly a power of two. Bounding the scale here is what keeps its
                 * reciprocal finite, and it costs nothing at the bottom of the range: a
                 * denormal magnitude scaled up by `1 / min()` lands somewhere its square is
                 * comfortably normal, which is the entire point - squaring it in place
                 * would have given zero.
                 *
                 * @return `2^(min_exponent - 1)`.
                 */
                [[nodiscard]] static T min_scale() noexcept { return std::numeric_limits<T>::min(); }

                /**
                 * @brief Largest value the scale is allowed to take.
                 *
                 * The reciprocal of @ref min_scale, and therefore itself a power of two
                 * whose own reciprocal is normal. In every IEEE format that lands one
                 * binade below the top, which is what makes `4 * max_scale()` overflow to
                 * infinity - the property @ref climb_to relies on to stop the scale
                 * climbing.
                 *
                 * @return `2^(max_exponent - 2)`.
                 */
                [[nodiscard]] static T max_scale() noexcept { return T(1) / std::numeric_limits<T>::min(); }

                /** @brief The value an empty norm has. */
                [[nodiscard]] static constexpr T identity() noexcept { return T(0); }

                /**
                 * @brief Combines two finished partial norms.
                 *
                 * `hypot`, not `sqrt(a*a + b*b)`, for the same reason this class exists at
                 * all: the squares are what overflow, and `hypot` is specified to compute
                 * the result without forming them. It is called once per tree level and
                 * once per work-group partial, never per element, so its cost is nothing.
                 *
                 * @param a The left partial norm.
                 * @param b The right partial norm.
                 * @return `sqrt(a^2 + b^2)`, computed safely.
                 */
                [[nodiscard]] static T combine(T a, T b) noexcept { return sycl::hypot(a, b); }

                /**
                 * @brief Folds one value into the running norm.
                 *
                 * Accepts either a raw element or an already-finished partial norm; the two
                 * are the same operation, which is what lets stage two of @ref reduce fold
                 * partials with the same accumulator that stage one folds elements with.
                 *
                 * @param x The value to fold in.
                 */
                void accumulate(T x) noexcept
                {
                    const T a = sycl::fabs(x);

                    // An exact zero contributes nothing, and letting it reach @ref climb_to
                    // would send the ladder there all the way to the bottom of the range for
                    // no gain.
                    if (a == T(0)) return;

                    // Leaving the current binade, not merely beating the running maximum.
                    // `hi_` is `2 * scale_`, so on a well-conditioned vector this fires a
                    // handful of times in total rather than once per element - and a NaN
                    // fails it, which is how a NaN reaches the multiply below unrescaled and
                    // propagates the way BLAS says it should.
                    if (a >= hi_) climb_to(a);

                    // Exact: `inv_scale_` is a power of two, and the bounds @ref climb_to
                    // keeps guarantee the product neither overflows nor underflows.
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
                 * @brief Raises the scale to @p a and rescales the total already folded to
                 * match.
                 *
                 * **Straight-line on purpose.** Two earlier versions of this rounded the
                 * scale down to an exact power of two, so that the rescale below and the
                 * per-element multiply were both exact. Both were slower than the whole
                 * scaling form needs to be, and by a lot, because both put something in the
                 * loop body that a compiler will not look past: `ldexp(1, ilogb(a))` puts
                 * two library calls there, and a doubling ladder puts two data-dependent
                 * loops. Measured over 100 launches of a 1M-element fp32 `nrm2` on this
                 * project's OpenCL CPU device - the branch that reaches here is taken a
                 * handful of times in each of those launches:
                 *
                 * | form                                   |   best |
                 * |----------------------------------------|-------:|
                 * | naive `sqrt(sum(x*x))` (wrong)          | 5.9 ms |
                 * | scaled, `ldexp`/`ilogb` binade          |14.7 ms |
                 * | scaled, doubling-ladder binade          |21.2 ms |
                 * | scaled, this                            |11.0 ms |
                 *
                 * **And exactness was not worth buying.** Rounding to a binade bought an
                 * exact `ssq_.rescale`, but every rescale shrinks the total already folded
                 * by at least four - that is what @ref hi_ guarantees - so a segment's
                 * rounding error is down-weighted exactly as fast as the segment is. The
                 * errors do not accumulate over the ~250 rescales a full-range vector can
                 * provoke; only the last few contribute, at a few ulp. The reciprocal is
                 * likewise a single systematic `(1 + eps)` on the whole answer, because
                 * every element in a segment is multiplied by the same rounded `1 / scale`.
                 *
                 * @param a A magnitude strictly above zero and at or above the current
                 *        @ref hi_. Never a NaN - the `>=` in @ref accumulate excludes one.
                 */
                void climb_to(T a) noexcept
                {
                    const T ceiling = max_scale();

                    // Clamped so the reciprocal below is a normal number. Neither end costs
                    // anything: a magnitude under the floor still scales to something whose
                    // square is normal, and one over the ceiling still scales to at most 4.
                    // An infinity lands exactly on the ceiling, which is what makes it
                    // scale to infinity and saturate the sum rather than produce a NaN.
                    T next = a;
                    if (next < min_scale()) next = min_scale();
                    if (next > ceiling) next = ceiling;

                    const T inv_next = T(1) / next;

                    // Zero on the very first value folded, which correctly clears an empty
                    // sum rather than dividing by a scale that does not exist yet.
                    const T shrink = scale_ * inv_next;
                    ssq_.rescale(shrink * shrink);

                    scale_ = next;
                    inv_scale_ = inv_next;

                    // 2 * next normally. At the ceiling `4 * next` overflows to infinity by
                    // construction - the ceiling is one binade below the top - which is what
                    // stops a vector of huge values from re-entering this once per element.
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
         * @class RunningProduct
         * @brief A running product, for the one folding reduction that is not a sum.
         *
         * No compensation, because there is nothing to compensate: Neumaier recovers
         * the bits an *addition* rounds away, and multiplication's rounding error is
         * not of that shape. What this shares with @ref CompensatedSum, and the only
         * reason it is spelled the same way, is the fixed combination order - which is
         * what the determinism argument actually rests on.
         *
         * @tparam T The element type.
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
         * @class RunningMax
         * @brief A running maximum, in the shape the folding seams accumulate in.
         *
         * **Not what the whole-tensor `max` uses, and that is deliberate.**
         * `ReductionOps::max` stays on `sycl::reduction`, because `maximum` is
         * exactly associative and exactly commutative, so every combination order
         * yields the same value and a hand-written fixed-order tree there would buy
         * nothing but slower code - the argument is written out in `max.cpp`. This
         * class exists because the *segmented* seam
         * (`Dispatch::execute_folding_axis_reduction`) is one shape serving every
         * axis reduction, and a shape needs an accumulator. Order-independence means
         * that costs nothing here either: the fold is fixed anyway, so a per-axis
         * maximum is bit-reproducible for a stronger reason than the additive
         * reductions are.
         *
         * **Selection matches `sycl::maximum` exactly**, down to which operand
         * survives: both keep the left one unless the right compares greater, so a
         * per-axis maximum and a whole-tensor maximum cannot disagree on the same
         * data. That carries over `max.cpp`'s one caveat as well - every comparison
         * against a NaN is false, so a NaN in the input does not propagate and which
         * value survives meeting one depends on the order. No reduction shape fixes
         * that; only screening the input does.
         *
         * @tparam T The element type.
         */
        template<typename T>
        class RunningMax
        {
            public:
                /**
                 * @brief The value an empty maximum has.
                 *
                 * The lowest finite value of @p T, so folding it in against anything
                 * representable is a no-op. Not `constexpr`, because
                 * `std::numeric_limits<sycl::half>::lowest()` is not - the same
                 * reason @ref ScaledSumOfSquares::min_scale is a function.
                 *
                 * @return `std::numeric_limits<T>::lowest()`.
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
         * @brief Reduces `in[0..n)` to `*out`, deterministically.
         *
         * Computes `finalize(fold(transform(in[i])), n)`, where `fold` is @p Accum's
         * operation applied in an order fixed entirely by the launch geometry. Submits
         * one kernel on the single-pass shape and two on the two-stage one; see
         * @ref plan for which, and the file header for why the order is nailed down.
         *
         * @tparam Accum The accumulation policy, @ref CompensatedSum or
         *         @ref RunningProduct. A class template on the element type, supplying
         *         `identity()`, `combine(a, b)`, `accumulate(x)` and `value()`.
         * @tparam T The element type, deduced.
         * @tparam Transform The per-element transform type, deduced.
         * @tparam Finalize The finalizer type, deduced.
         * @param queue The queue to submit to.
         * @param in The input array.
         * @param out Where to store the single result.
         * @param n How many elements to reduce.
         * @param partials Scratch for one value per work-group; may be null when
         *        @p partial_capacity is zero.
         * @param partial_capacity How many values @p partials can hold.
         * @param transform Callable `(T) -> T`, applied to each element before it is
         *        folded in. `x * x` for a 2-norm, `fabs(x)` for an absolute sum.
         * @param finalize Callable `(T value, std::size_t n) -> T`, applied once to the
         *        folded result. `sqrt` for a 2-norm, `/n` for a mean. Folding it in here
         *        rather than submitting a second kernel is what lets `mean` and `nrm2`
         *        cost the same number of launches as `sum`.
         * @param deps Events this submission must depend on.
         * @return The event for the last submitted kernel.
         *
         * @warning @p transform and @p finalize are inlined into a SYCL device kernel:
         *          they must be device-copyable, capture by value, hold no host
         *          pointers, allocate nothing and throw nothing.
         */
        template<template<typename> class Accum, typename T,
                 typename Transform, typename Finalize>
        [[nodiscard]] sycl::event reduce(sycl::queue& queue, const T* in, T* out,
                                         std::size_t n, T* partials,
                                         std::size_t partial_capacity,
                                         Transform transform, Finalize finalize,
                                         const std::vector<sycl::event>& deps)
        {
            const Plan geometry = plan(Device::profile_for(queue), n, partial_capacity);

            if (geometry.sequential())
            {
                // One task, one ascending walk. Worth being precise about what that does
                // and does not promise: it fixes the order *this code* asks for, not
                // necessarily the order the machine executes. Removing the compensation
                // from CompensatedSum and re-running this path on the OpenCL CPU device
                // produced 100003840 where a strict element-by-element float sum gives
                // exactly 100000000 - the device JIT had vectorised the loop and folded
                // the lanes, which is a reassociation. With the compensation in place it
                // does not, and either way the same binary on the same device produces
                // the same bits on every run, which is the guarantee that was asked for.
                // So "sequential" here means "no cross-work-item combination this code
                // has to pin down", not "provably index order all the way to the silicon".
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        Accum<T> acc;
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

                sycl::local_accessor<T, 1> tile(sycl::range<1>(local), h);

                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local),
                                                 sycl::range<1>(local)),
                               [=](sycl::nd_item<1> it)
                {
                    const std::size_t gid = it.get_global_id(0);
                    const std::size_t lid = it.get_local_id(0);
                    const std::size_t stride = groups * local;

                    // A grid-strided slice: work-item g owns indices g, g + stride,
                    // g + 2*stride, ... Which indices those are, and the order they are
                    // visited in, come only from the geometry, so they are the same on
                    // every run. Consecutive items also read consecutive addresses at
                    // every step, which is what keeps the loads coalesced.
                    Accum<T> acc;
                    for (std::size_t i = gid; i < n; i += stride)
                        acc.accumulate(transform(in[i]));

                    tile[lid] = acc.value();

                    sycl::group_barrier(it.get_group());

                    // A hand-written halving tree, and NOT sycl::reduce_over_group. The
                    // spec permits that function to combine in any order it likes, which
                    // for a non-associative operator means it may return different bits
                    // on different launches of the same kernel. This loop combines
                    // (lid, lid + off) for a fixed descending sequence of offsets, so
                    // the tree is the same tree every time. `local` is a power of two by
                    // construction (see floor_pow2), which is what makes the halving
                    // visit every lane exactly once.
                    for (std::size_t off = local / 2; off > 0; off >>= 1)
                    {
                        if (lid < off) tile[lid] = Accum<T>::combine(tile[lid], tile[lid + off]);
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
                    // Ascending index order over a count fixed at launch. Compensated
                    // as well, which costs nothing at this length and recovers the
                    // spread between partials that a plain walk would drop.
                    Accum<T> acc;
                    for (std::size_t g = 0; g < groups; ++g) acc.accumulate(partials[g]);
                    *out = finalize(acc.value(), n);
                });
            });
        }

    } // namespace Reduce
} // namespace SushiBLAS
