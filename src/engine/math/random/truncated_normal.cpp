/**************************************************************************/
/* truncated_normal.cpp                                                   */
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
 * @file truncated_normal.cpp
 * @brief Defines RandomOps::truncated_normal, which records the truncated normal fill of a tensor.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <limits>
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
        /** @brief Reports whether Phi(@p z) is below the smallest normal T. */
        template<typename T>
        inline bool is_massless_below(double z)
        {
            return Kernels::SYCL::Random::standard_normal_cdf(static_cast<T>(z))
                   < std::numeric_limits<T>::min();
        }
    } // namespace

    TaskHandle RandomOps::truncated_normal(Tensor& t, double mean, double stddev, double a,
                                           double b)
    {
        recorder_.require_sycl_path("RNG operation", "random.truncated_normal");
        Dispatch::require_operand(std::isfinite(a) && std::isfinite(b) && a < b,
                                  "random.truncated_normal", "a",
                                  "and b are {} and {}; truncated_normal needs finite bounds "
                                  "with a < b",
                                  a, b);
        Dispatch::require_random_dtype(t, Dispatch::Domain::REAL, Dispatch::Sampling::DRAWS_VALUES,
                                       "random.truncated_normal");

        constexpr double k_bound = Kernels::SYCL::Random::K_TRUNCATED_NORMAL_MAX_BOUND;
        const double a_clamped = std::max(a, -k_bound);
        const double b_clamped = std::min(b, k_bound);

        // An interval above zero is drawn as its mirror image below zero and negated.
        const bool mirrored = a_clamped > 0.0;
        const double lo = mirrored ? -b_clamped : a_clamped;
        const double hi = mirrored ? -a_clamped : b_clamped;
        const double scale = mirrored ? -stddev : stddev;

        // hi is the bound nearer the mean, so no mass below hi means no mass in the interval.
        const bool massless = t.dtype == Core::DataType::FLOAT32 ? is_massless_below<float>(hi)
                                                                 : is_massless_below<double>(hi);
        Dispatch::require_operand(!massless, "random.truncated_normal", "a",
                                  "and b are {} and {}; in dtype {} the normal distribution "
                                  "function rounds to zero between them",
                                  a, b, Core::to_string(t.dtype));

        return Dispatch::execute_random<Dispatch::Domain::REAL>(
            recorder_, rng_, t, "random.truncated_normal", "random.truncated_normal"_op,
            {mean, stddev, a_clamped, b_clamped},
            [mean, scale, lo, hi](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset,
                                  int64_t size, auto* pT,
                                  const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                const T m = static_cast<T>(mean), sd = static_cast<T>(scale);
                const T z_lo = static_cast<T>(lo), z_hi = static_cast<T>(hi);

                return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t block)
                {
                    using Kernels::SYCL::Random::draw_truncated_standard_normal;
                    Kernels::SYCL::Random::DrawBlock<T> values =
                        draw_truncated_standard_normal<T>(seed, offset, block, z_lo, z_hi);
                    for (T& value : values)
                        value = m + sd * value;
                    return values;
                });
            });
    }
} // namespace SushiBLAS
