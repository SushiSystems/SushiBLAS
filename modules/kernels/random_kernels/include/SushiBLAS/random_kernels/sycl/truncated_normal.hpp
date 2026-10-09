/**************************************************************************/
/* truncated_normal.hpp                                                   */
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
 * @file truncated_normal.hpp
 * @brief Defines the truncated standard normal draw by bisection of the distribution function.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/sycl/philox.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Random
            {
                /** @brief Holds one over the square root of two, which scales erfc to Phi. */
                inline constexpr double K_INVERSE_SQRT_TWO = 0.70710678118654752440;

                /**
                 * @brief Holds the largest standardised bound magnitude the bisection receives.
                 * @see kernels/README.md
                 */
                inline constexpr double K_TRUNCATED_NORMAL_MAX_BOUND = 40.0;

                /** @brief Returns the standard normal distribution function at @p z. */
                template<typename T>
                inline T standard_normal_cdf(T z)
                {
                    const T scaled = -z * static_cast<T>(K_INVERSE_SQRT_TWO);
                    return static_cast<T>(0.5) * sycl::erfc(scaled);
                }

                /**
                 * @brief Draws K_DRAWS_PER_BLOCK standard normals inside [lo, hi] by bisection.
                 * @pre lo < hi, both within K_TRUNCATED_NORMAL_MAX_BOUND, and lo <= 0.
                 * @see kernels/README.md
                 */
                template<typename T>
                inline DrawBlock<T> draw_truncated_standard_normal(uint64_t seed, uint64_t offset,
                                                                   uint64_t block, T lo, T hi)
                {
                    DrawBlock<T> values = uniform_block<T>(seed, offset, block,
                                                           standard_normal_cdf(lo),
                                                           standard_normal_cdf(hi));

                    for (T& value : values)
                    {
                        const T target = value;
                        T below = lo;
                        T above = hi;

                        // below and above only close in, so the result stays in [lo, hi].
                        for (int step = 0; step < std::numeric_limits<T>::digits; ++step)
                        {
                            const T mid = below + (above - below) * static_cast<T>(0.5);

                            if (standard_normal_cdf(mid) < target)
                                below = mid;
                            else
                                above = mid;
                        }

                        value = below + (above - below) * static_cast<T>(0.5);
                    }

                    return values;
                }
            } // namespace Random
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
