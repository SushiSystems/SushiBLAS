/****************************************************************/
/* truncated_normal.cpp                                         */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file truncated_normal.cpp
 * @brief Defines RandomOps::truncated_normal, which records the truncated normal fill of a tensor.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Holds one over the square root of two, which scales erfc to the normal law. */
        constexpr double K_INVERSE_SQRT_TWO = 0.70710678118654752440;

        /** @brief Returns the standard normal distribution function at @p z. */
        template<typename T>
        inline T standard_normal_cdf(T z)
        {
            return static_cast<T>(0.5) * sycl::erfc(-z * static_cast<T>(K_INVERSE_SQRT_TWO));
        }

        /** @brief Reports whether the distribution function at @p z is below the smallest normal T. */
        template<typename T>
        inline bool is_massless_below(double z)
        {
            return standard_normal_cdf(static_cast<T>(z)) < std::numeric_limits<T>::min();
        }

        /**
         * @brief Draws one standard normal value inside [lo, hi] by bisecting its distribution.
         * @pre lo < hi, both finite, and lo <= 0 so the distribution function keeps its precision.
         */
        template<typename T>
        inline T draw_truncated_standard_normal(uint64_t seed, uint64_t offset, uint64_t index, T lo, T hi)
        {
            const T target = Kernels::SYCL::Random::draw_uniform<T>(seed, offset, index,
                                                                    standard_normal_cdf(lo),
                                                                    standard_normal_cdf(hi));

            // lo and hi only move toward each other, so the result never leaves [lo, hi].
            for (int step = 0; step < std::numeric_limits<T>::digits; ++step)
            {
                const T mid = lo + (hi - lo) * static_cast<T>(0.5);

                if (standard_normal_cdf(mid) < target)
                    lo = mid;
                else
                    hi = mid;
            }

            return lo + (hi - lo) * static_cast<T>(0.5);
        }
    } // namespace

    sycl::event RandomOps::truncated_normal(Tensor& t, double mean, double stddev, double a, double b)
    {
        SB_THROW_IF(!std::isfinite(a) || !std::isfinite(b) || !(a < b),
                    "RandomOps::truncated_normal needs finite bounds with a < b, got a = {} and b = {}.",
                    a, b);
        SB_THROW_IF(Core::is_complex_dtype(t.dtype),
                    "RandomOps::truncated_normal does not support dtype {}: a complex value has "
                    "no order to bound.",
                    Core::to_string(t.dtype));

        // An interval above zero is drawn as its mirror image below zero and negated.
        const bool mirrored = a > 0.0;
        const double lo = mirrored ? -b : a;
        const double hi = mirrored ? -a : b;
        const double scale = mirrored ? -stddev : stddev;

        // hi is the bound nearer the mean, so no mass below hi means no mass in the interval.
        const bool massless = t.dtype == Core::DataType::FLOAT32 ? is_massless_below<float>(hi)
                                                                 : is_massless_below<double>(hi);
        SB_THROW_IF(massless,
                    "RandomOps::truncated_normal cannot draw between a = {} and b = {} in dtype {}: "
                    "the normal distribution function rounds to zero there.",
                    a, b, Core::to_string(t.dtype));

        return Dispatch::execute_random(engine_, t, "random.truncated_normal", "random.truncated_normal"_op, {mean, stddev, a, b},
            [mean, scale, lo, hi](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if constexpr (Core::is_complex_v<T>)
                {
                    // Complex dtypes are rejected at record time, so no task reaches this branch.
                    return sycl::event();
                }
                else
                {
                    T m = static_cast<T>(mean), sd = static_cast<T>(scale);
                    T z_lo = static_cast<T>(lo), z_hi = static_cast<T>(hi);

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        return m + sd * draw_truncated_standard_normal<T>(seed, offset, i, z_lo, z_hi);
                    });
                }
            });
    }
} // namespace SushiBLAS
