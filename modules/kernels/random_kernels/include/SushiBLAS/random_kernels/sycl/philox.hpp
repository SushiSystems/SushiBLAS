/**************************************************************************/
/* philox.hpp                                                             */
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
/*                                                                        */
/* Philox as in Random123, Copyright 2010-2012, D. E. Shaw Research       */
/* BSD-3-Clause, see NOTICE.md                                            */
/**************************************************************************/

/**
 * @file philox.hpp
 * @brief Defines Philox4x32-10 and the uniform, normal and index draws built on it.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Random
            {
                /** @brief Holds the 32-bit multiply one Philox round is built from. */
                namespace Detail
                {
                    /** @brief Splits the 64-bit product of @p a and @p b into @p hi and @p lo. */
                    inline void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo)
                    {
                        uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
                        hi = static_cast<uint32_t>(product >> 32);
                        lo = static_cast<uint32_t>(product);
                    }
                } // namespace Detail

                /**
                 * @brief Returns the four words Philox4x32-10 maps the counter (c0..c3) to.
                 * @see kernels/README.md
                 */
                inline std::array<uint32_t, 4> philox4x32_10(uint32_t key0, uint32_t key1,
                                                              uint32_t c0, uint32_t c1,
                                                              uint32_t c2, uint32_t c3)
                {
                    constexpr uint32_t M0 = 0xD2511F53u;
                    constexpr uint32_t M1 = 0xCD9E8D57u;
                    constexpr uint32_t W0 = 0x9E3779B9u;
                    constexpr uint32_t W1 = 0xBB67AE85u;

                    for (int round = 0; round < 10; ++round)
                    {
                        uint32_t hi0, lo0, hi1, lo1;
                        Detail::mulhilo32(M0, c0, hi0, lo0);
                        Detail::mulhilo32(M1, c2, hi1, lo1);

                        uint32_t next_c0 = hi1 ^ c1 ^ key0;
                        uint32_t next_c1 = lo1;
                        uint32_t next_c2 = hi0 ^ c3 ^ key1;
                        uint32_t next_c3 = lo0;

                        c0 = next_c0; c1 = next_c1; c2 = next_c2; c3 = next_c3;
                        key0 += W0; key1 += W1;
                    }

                    return {c0, c1, c2, c3};
                }

                /** @brief Returns the four words of block @p block of the (seed, offset) Philox stream. */
                inline std::array<uint32_t, 4> philox_block(uint64_t seed, uint64_t offset, uint64_t block)
                {
                    return philox4x32_10(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                                         static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32),
                                         static_cast<uint32_t>(block), static_cast<uint32_t>(block >> 32));
                }

                /** @brief Maps the top 24 bits of a word to [0, 1); 1 - 2^-24 is the largest result. */
                inline float uniform01_float(uint32_t w)
                {
                    return static_cast<float>(w >> 8) * (1.0f / 16777216.0f);
                }

                /** @brief Maps the top 53 bits of two words to [0, 1); 1 - 2^-53 is the largest result. */
                inline double uniform01_double(uint32_t w_hi, uint32_t w_lo)
                {
                    uint64_t combined = (static_cast<uint64_t>(w_hi) << 32) | static_cast<uint64_t>(w_lo);
                    return static_cast<double>(combined >> 11) * (1.0 / 9007199254740992.0);
                }

                /**
                 * @brief Holds how many draws of T one Philox block yields: four floats or two doubles.
                 * @see kernels/README.md
                 */
                template<typename T>
                inline constexpr std::size_t K_DRAWS_PER_BLOCK = std::is_same_v<T, double> ? 2 : 4;

                /** @brief Holds the draws of T that one Philox block yields. */
                template<typename T>
                using DrawBlock = std::array<T, K_DRAWS_PER_BLOCK<T>>;

                /** @brief Returns every uniform draw in [0, 1) that the words of one Philox block hold. */
                template<typename T>
                inline DrawBlock<T> uniform01_block(const std::array<uint32_t, 4>& w)
                {
                    DrawBlock<T> u{};
                    if constexpr (std::is_same_v<T, double>)
                    {
                        u[0] = uniform01_double(w[1], w[0]);
                        u[1] = uniform01_double(w[3], w[2]);
                    }
                    else
                    {
                        for (std::size_t k = 0; k < 4; ++k)
                            u[k] = static_cast<T>(uniform01_float(w[k]));
                    }
                    return u;
                }

                /**
                 * @brief Draws K_DRAWS_PER_BLOCK uniform values in [lo, hi) from Philox block @p block.
                 * @pre lo <= hi, both finite, and hi - lo finite; lo == hi fills lo.
                 * @see kernels/README.md
                 */
                template<typename T>
                inline DrawBlock<T> uniform_block(uint64_t seed, uint64_t offset, uint64_t block, T lo, T hi)
                {
                    DrawBlock<T> values = uniform01_block<T>(philox_block(seed, offset, block));
                    for (T& value : values)
                    {
                        const T scaled = lo + value * (hi - lo);
                        // Rounding of lo + u * (hi - lo) can reach hi although u < 1.
                        value = scaled < hi ? scaled : sycl::nextafter(hi, lo);
                    }
                    return values;
                }

                /**
                 * @brief Draws K_DRAWS_PER_BLOCK normal values from Philox block @p block by Box-Muller.
                 * @see kernels/README.md
                 */
                template<typename T>
                inline DrawBlock<T> normal_block(uint64_t seed, uint64_t offset, uint64_t block, T mean, T stddev)
                {
                    constexpr double TWO_PI = 6.283185307179586476925286766559;

                    const DrawBlock<T> u = uniform01_block<T>(philox_block(seed, offset, block));
                    DrawBlock<T> values{};
                    for (std::size_t pair = 0; pair < K_DRAWS_PER_BLOCK<T>; pair += 2)
                    {
                        // u lies in [0, 1), so 1 - u lies in (0, 1] and its logarithm is finite.
                        const T r = sycl::sqrt(static_cast<T>(-2.0) * sycl::log(static_cast<T>(1.0) - u[pair]));
                        const T theta = static_cast<T>(TWO_PI) * u[pair + 1];
                        values[pair] = mean + stddev * r * sycl::cos(theta);
                        values[pair + 1] = mean + stddev * r * sycl::sin(theta);
                    }
                    return values;
                }

                /**
                 * @brief Draws one integer in [0, bound) as a 64-bit draw modulo bound.
                 * @see kernels/README.md
                 */
                inline uint64_t draw_index(uint64_t seed, uint64_t offset, uint64_t index, uint64_t bound)
                {
                    if (bound <= 1)
                        return 0;

                    auto w = philox_block(seed, offset, index);
                    uint64_t raw = (static_cast<uint64_t>(w[1]) << 32) | static_cast<uint64_t>(w[0]);
                    return raw % bound;
                }

            } // namespace Random
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
