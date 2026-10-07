/**************************************************************************/
/* random.hpp                                                             */
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
 * @file random.hpp
 * @brief Declares RandomOps, the random fills and initialisers.
 * @author Mustafa Garip
 */

#pragma once

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @brief Holds the largest lambda RandomOps::poisson accepts.
     * @see include/SushiBLAS/engine/math/README.md
     */
    inline constexpr double K_MAX_POISSON_LAMBDA = 1.0e7;

    /**
     * @class RandomOps
     * @brief Random number generators for tensor initialization and statistics.
     * 
     * Uses a hand-written, portable Philox4x32-10 counter-based PRNG implemented
     * directly in SYCL to generate numbers from various distributions on the
     * accelerator - no oneMKL dependency.
     */
    class RandomOps 
    {
        public:
            explicit RandomOps(Engine& e) : engine_(e) {}

            /**
             * @brief Set the seed for this engine's random number generation.
             * @param seed The seed value.
             */
            void set_seed(uint64_t seed);

            /**
             * @brief Alias for set_seed.
             */
            inline void seed(uint64_t s) { set_seed(s); }

            /**
             * @brief Fills a tensor with a constant value.
             * @param t Tensor to fill.
             * @param value The constant value.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event constant(Tensor& t, double value);

            /**
             * @brief Fills a tensor from a uniform distribution on [min, max).
             * @param t Tensor to fill; FLOAT32, FLOAT64, COMPLEX32 or COMPLEX64, contiguous.
             * @param min Minimum value (default 0.0); finite, not above max; min == max fills min.
             * @param max Maximum value (default 1.0); max - min finite in t's dtype.
             * @throws std::invalid_argument unless the bounds are finite and ordered.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event uniform(Tensor& t, double min = 0.0, double max = 1.0);

            /**
             * @brief Fills a tensor from a normal distribution.
             * @param mean Mean of the distribution; finite.
             * @param stddev Standard deviation; finite and at least 0.
             * @throws std::invalid_argument when mean or stddev breaks its rule, or t is HALF.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event normal(Tensor& t, double mean = 0.0, double stddev = 1.0);

            /**
             * @brief Fills a tensor from a log-normal distribution.
             * @param mean Mean of the underlying normal distribution; finite.
             * @param stddev Standard deviation of the underlying normal; finite and at least 0.
             * @throws std::invalid_argument when mean or stddev breaks its rule, or t is not real.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event log_normal(Tensor& t, double mean = 0.0, double stddev = 1.0);

            /**
             * @brief Fills a tensor from an exponential distribution.
             * @param lambda Rate, one over the mean; finite and above 0.
             * @throws std::invalid_argument when lambda breaks its rule or t is not real.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event exponential(Tensor& t, double lambda = 1.0);

            /**
             * @brief Fills a tensor from a Poisson distribution, by Knuth below 10 and PTRS above.
             * @param t Tensor to fill; FLOAT32 or FLOAT64, contiguous, on a device with fp64.
             * @param lambda Mean of the distribution; finite and in [0, K_MAX_POISSON_LAMBDA].
             * @throws std::invalid_argument when lambda or t breaks its rule.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event poisson(Tensor& t, double lambda = 1.0);

            /**
             * @brief Fills a tensor with integers drawn uniformly from the closed range [min, max].
             * @param t Tensor to fill; FLOAT64, or FLOAT32 while |min| and |max| are at most 2^24.
             * @param max Maximum value (inclusive); not below min.
             * @throws std::invalid_argument when min > max, t is complex or FLOAT32 past 2^24.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event discrete_uniform(Tensor& t, int32_t min, int32_t max);

            /**
             * @brief Fills a tensor from a truncated normal distribution.
             * @param a Lower bound, in standard deviations; clamped to -40 before recording.
             * @param b Upper bound, in standard deviations; clamped to 40 before recording.
             * @throws std::invalid_argument when t is not FLOAT32 or FLOAT64.
             * @throws std::runtime_error when a and b are not finite with a < b, or bound no mass in t's dtype.
             */
            sycl::event truncated_normal(Tensor& t, double mean = 0.0, double stddev = 1.0,
                                         double a = -2.0, double b = 2.0);

            /**
             * @brief Fills a tensor from a Bernoulli distribution.
             * @param t Tensor to fill; FLOAT32 or FLOAT64, contiguous.
             * @param p Probability of a 1; within [0, 1].
             * @throws std::invalid_argument when p is NaN or outside [0, 1], or t is complex.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event bernoulli(Tensor& t, double p = 0.5);

            /**
             * @brief Fills a tensor by Xavier (Glorot) uniform initialisation.
             * @throws std::invalid_argument when n_in + n_out is not above 0 or t is not real.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event xavier_uniform(Tensor& t, int64_t n_in, int64_t n_out);

            /**
             * @brief Fills a tensor by Xavier (Glorot) normal initialisation.
             * @throws std::invalid_argument when n_in + n_out is not above 0 or t is not real.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event xavier_normal(Tensor& t, int64_t n_in, int64_t n_out);

            /**
             * @brief Fills a tensor by He (Kaiming) uniform initialisation.
             * @throws std::invalid_argument when n_in is not above 0 or t is not real.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event he_uniform(Tensor& t, int64_t n_in);

            /**
             * @brief Fills a tensor by He (Kaiming) normal initialisation.
             * @throws std::invalid_argument when n_in is not above 0 or t is not real.
             * @return A default-constructed event; the fill runs at Engine::execute.
             */
            sycl::event he_normal(Tensor& t, int64_t n_in);

            /**
             * @brief Fills a matrix with a random (semi-)orthogonal matrix scaled by gain.
             * @param t Must be rank 2, non-empty, and FLOAT32 or FLOAT64.
             * @throws std::invalid_argument when t is not FLOAT32 or FLOAT64, has no storage or
             * is not contiguous.
             * @throws std::runtime_error when t is not a non-empty rank-2 tensor.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event orthogonal(Tensor& t, double gain = 1.0);

            /**
             * @brief Permutes the elements of a tensor in place with a Fisher-Yates shuffle.
             * @throws std::invalid_argument when t is INT32 or INT64.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event shuffle(Tensor& t);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
