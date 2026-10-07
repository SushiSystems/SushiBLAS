/**************************************************************************/
/* min.cpp                                                                */
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
 * @file min.cpp
 * @brief Defines ReductionOps::min, which records the minimum reduction.
 * @author Mustafa Garip
 */

#include <limits>
#include <cstddef>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Holds a running minimum that turns into NaN once any NaN is folded.
         * @see src/README.md
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

                /**
                 * @brief Folds one value into the running minimum.
                 * @param x The value to consider.
                 */
                void accumulate(T x) noexcept
                {
                    best_ = combine(best_, x);
                }

                /**
                 * @brief Returns the minimum so far, NaN if a NaN was folded, or identity() if
                 *        nothing was.
                 */
                [[nodiscard]] T value() const noexcept
                {
                    return best_;
                }

            private:
                /** @brief The smaller value folded so far, or the first NaN met. */
                T best_ = identity();
        };
    } // namespace

    sycl::event ReductionOps::min(const Tensor& t, Tensor& result)
    {
        Dispatch::require_storage(t, "math.min", "t");
        const std::size_t partials = Dispatch::planned_partials_for(t);

        auto dispatch = [](sycl::queue& q, const auto* pT, auto* pR, int64_t size,
                           auto scratch, const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;

            const T init_val = std::numeric_limits<T>::max();

            // An empty input writes the max() of the element type itself, not of its accumulation type.
            if (size == 0)
            {
                return q.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]() { *pR = init_val; });
                });
            }

            return Reduce::reduce<NaNPropagatingMin>(q, pT, pR, static_cast<std::size_t>(size),
                                              scratch.data, scratch.capacity,
                                              Dispatch::ReduceIdentityTransform{},
                                              Dispatch::ReduceIdentityFinalize{}, deps);
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.min", "math.min"_op,
                                           dispatch, partials);
    }
} // namespace SushiBLAS
