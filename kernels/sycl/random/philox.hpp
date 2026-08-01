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

#include <array>
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
                /**
                * @brief Philox4x32-10, a counter-based PRNG (Salmon, Moraes, Dror, Shaw,
                * "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11). Stateless: the same
                * (key, counter) pair always produces the same 4-word output, so any element
                * index can be drawn independently and in parallel with no shared RNG state.
                * Round constants verified against the reference Random123 implementation
                * (DEShawResearch/random123, include/Random123/philox.h).
                *
                * This replaces oneMKL's philox4x32x10 (SB-2): the algorithm is unchanged,
                * only the backend is now hand-written SYCL with no oneMKL dependency.
                */
                namespace Detail
                {
                    inline void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo)
                    {
                        uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
                        hi = static_cast<uint32_t>(product >> 32);
                        lo = static_cast<uint32_t>(product);
                    }
                }

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

                // Maps a raw 32/64-bit word to the open interval (0,1), never exactly 0 or 1 -
                // callers (log_normal via Box-Muller, exponential's inverse-CDF) take log(u).
                inline float uniform01_float(uint32_t w)
                {
                    return (static_cast<float>(w) + 0.5f) * (1.0f / 4294967296.0f);
                }

                inline double uniform01_double(uint32_t w_hi, uint32_t w_lo)
                {
                    uint64_t combined = (static_cast<uint64_t>(w_hi) << 32) | static_cast<uint64_t>(w_lo);
                    return (static_cast<double>(combined) + 0.5) * (1.0 / 18446744073709551616.0);
                }

                template<typename T>
                inline T uniform01(const std::array<uint32_t, 4>& w, int pair)
                {
                    if constexpr (std::is_same_v<T, double>)
                        return static_cast<T>(uniform01_double(w[pair * 2 + 1], w[pair * 2]));
                    else
                        return static_cast<T>(uniform01_float(w[pair * 2]));
                }

                // One Philox call per element: key = (seed_lo, seed_hi), counter = (offset_lo,
                // offset_hi, index_lo, index_hi). Keying the counter's high word on the
                // per-task `offset` (rather than offset*size, as the old oneMKL skip_ahead call
                // did) means two random-generating tasks never share a (key, counter) pair
                // regardless of their tensor sizes.
                template<typename T>
                inline T draw_uniform(uint64_t seed, uint64_t offset, uint64_t index, T lo, T hi)
                {
                    auto w = philox4x32_10(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                                           static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32),
                                           static_cast<uint32_t>(index), static_cast<uint32_t>(index >> 32));
                    T u = uniform01<T>(w, 0);
                    return lo + u * (hi - lo);
                }

                // Box-Muller: one Philox call yields both uniforms needed for one Gaussian
                // sample (word pair 0 -> u1, word pair 1 -> u2). Only the cosine branch (z0) is
                // used, leaving z1 unused - simplicity over throughput for this first pass.
                template<typename T>
                inline T draw_normal(uint64_t seed, uint64_t offset, uint64_t index, T mean, T stddev)
                {
                    // MSVC only defines M_PI with _USE_MATH_DEFINES, so spell it out instead.
                    constexpr double TWO_PI = 6.283185307179586476925286766559;

                    auto w = philox4x32_10(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                                           static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32),
                                           static_cast<uint32_t>(index), static_cast<uint32_t>(index >> 32));
                    T u1 = uniform01<T>(w, 0);
                    T u2 = uniform01<T>(w, 1);
                    T r = sycl::sqrt(static_cast<T>(-2.0) * sycl::log(u1));
                    T z0 = r * sycl::cos(static_cast<T>(TWO_PI) * u2);
                    return mean + stddev * z0;
                }

            } // namespace Random
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
