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

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the largest lambda RandomOps::poisson accepts.
     * @see modules/operations/MATH_README.md
     */
    inline constexpr double K_MAX_POISSON_LAMBDA = 1.0e7;

    /**
     * @class RandomOps
     * @brief Records random fills of tensors, drawing from one random stream.
     *
     * Uses a hand-written, portable Philox4x32-10 counter-based PRNG implemented
     * directly in SYCL to generate numbers from various distributions on the
     * accelerator - no oneMKL dependency.
     */
    class RandomOps final
    {
        public:
            /** @brief Builds a view over a recorder and a stream; both must outlive the view. */
            RandomOps(TaskRecorder& recorder, RNGStream& stream) noexcept
                : recorder_(recorder), rng_(stream)
            {

            }

            RandomOps(const RandomOps&) = default;
            RandomOps& operator=(const RandomOps&) = delete;
            RandomOps(RandomOps&&) = default;
            RandomOps& operator=(RandomOps&&) = delete;
            ~RandomOps() = default;

            /** @brief Sets the stream's seed and rewinds its offset to zero. */
            void set_seed(uint64_t seed);

            /**
             * @brief Alias for set_seed.
             */
            inline void seed(uint64_t s) { set_seed(s); }

            /**
             * @brief Fills a tensor with a constant value.
             * @param t Tensor to fill.
             * @param value The constant value.
             * @return The handle of the recorded task.
             */
            TaskHandle constant(Tensor& t, double value);

            /**
             * @brief Fills a tensor from a uniform distribution on [min, max).
             * @param t Tensor to fill; FLOAT32, FLOAT64, COMPLEX32 or COMPLEX64, contiguous.
             * @param min Minimum value (default 0.0); finite, not above max; min == max fills min.
             * @param max Maximum value (default 1.0); max - min finite in t's dtype.
             * @throws std::invalid_argument unless the bounds are finite and ordered.
             * @return The handle of the recorded task.
             */
            TaskHandle uniform(Tensor& t, double min = 0.0, double max = 1.0);

            /**
             * @brief Fills a tensor from a normal distribution.
             * @param mean Mean of the distribution; finite.
             * @param stddev Standard deviation; finite and at least 0.
             * @throws std::invalid_argument when mean or stddev breaks its rule, or t is HALF.
             * @return The handle of the recorded task.
             */
            TaskHandle normal(Tensor& t, double mean = 0.0, double stddev = 1.0);

            /**
             * @brief Fills a tensor from a log-normal distribution.
             * @param mean Mean of the underlying normal distribution; finite.
             * @param stddev Standard deviation of the underlying normal; finite and at least 0.
             * @throws std::invalid_argument when mean or stddev breaks its rule, or t is not real.
             * @return The handle of the recorded task.
             */
            TaskHandle log_normal(Tensor& t, double mean = 0.0, double stddev = 1.0);

            /**
             * @brief Fills a tensor from an exponential distribution.
             * @param lambda Rate, one over the mean; finite and above 0.
             * @throws std::invalid_argument when lambda breaks its rule or t is not real.
             * @return The handle of the recorded task.
             */
            TaskHandle exponential(Tensor& t, double lambda = 1.0);

            /**
             * @brief Fills a tensor from a Poisson distribution, by Knuth below 10 and PTRS above.
             * @param t Tensor to fill; FLOAT32 or FLOAT64, contiguous, on a device with fp64.
             * @param lambda Mean of the distribution; finite and in [0, K_MAX_POISSON_LAMBDA].
             * @throws std::invalid_argument when lambda or t breaks its rule.
             * @throws UnsupportedExecution when the device of t lacks the fp64 aspect.
             * @return The handle of the recorded task.
             */
            TaskHandle poisson(Tensor& t, double lambda = 1.0);

            /**
             * @brief Fills a tensor with integers drawn uniformly from the closed range [min, max].
             * @param t Tensor to fill; FLOAT64, or FLOAT32 while |min| and |max| are at most 2^24.
             * @param max Maximum value (inclusive); not below min.
             * @throws std::invalid_argument when min > max, t is complex or FLOAT32 past 2^24.
             * @return The handle of the recorded task.
             */
            TaskHandle discrete_uniform(Tensor& t, int32_t min, int32_t max);

            /**
             * @brief Fills a tensor from a truncated normal distribution.
             * @param a Lower bound, in standard deviations; clamped to -40 before recording.
             * @param b Upper bound, in standard deviations; clamped to 40 before recording.
             * @throws std::invalid_argument when t is not FLOAT32 or FLOAT64, when a and b are not
             *         finite with a < b, or when they bound no mass in t's dtype.
             */
            TaskHandle truncated_normal(Tensor& t, double mean = 0.0, double stddev = 1.0,
                                        double a = -2.0, double b = 2.0);

            /**
             * @brief Fills a tensor from a Bernoulli distribution.
             * @param t Tensor to fill; FLOAT32 or FLOAT64, contiguous.
             * @param p Probability of a 1; within [0, 1].
             * @throws std::invalid_argument when p is NaN or outside [0, 1], or t is complex.
             * @return The handle of the recorded task.
             */
            TaskHandle bernoulli(Tensor& t, double p = 0.5);

            /**
             * @brief Fills a tensor by Xavier (Glorot) uniform initialisation.
             * @throws std::invalid_argument when n_in + n_out is not above 0 or t is not real.
             * @return The handle of the recorded task.
             */
            TaskHandle xavier_uniform(Tensor& t, int64_t n_in, int64_t n_out);

            /**
             * @brief Fills a tensor by Xavier (Glorot) normal initialisation.
             * @throws std::invalid_argument when n_in + n_out is not above 0 or t is not real.
             * @return The handle of the recorded task.
             */
            TaskHandle xavier_normal(Tensor& t, int64_t n_in, int64_t n_out);

            /**
             * @brief Fills a tensor by He (Kaiming) uniform initialisation.
             * @throws std::invalid_argument when n_in is not above 0 or t is not real.
             * @return The handle of the recorded task.
             */
            TaskHandle he_uniform(Tensor& t, int64_t n_in);

            /**
             * @brief Fills a tensor by He (Kaiming) normal initialisation.
             * @throws std::invalid_argument when n_in is not above 0 or t is not real.
             * @return The handle of the recorded task.
             */
            TaskHandle he_normal(Tensor& t, int64_t n_in);

            /**
             * @brief Fills a matrix with a random (semi-)orthogonal matrix scaled by gain.
             * @param t Must be rank 2, non-empty, and FLOAT32 or FLOAT64.
             * @throws std::invalid_argument when t is not a non-empty rank-2 FLOAT32 or FLOAT64
             *         tensor, has no storage or is not contiguous.
             * @see modules/operations/MATH_README.md
             */
            TaskHandle orthogonal(Tensor& t, double gain = 1.0);

            /**
             * @brief Permutes the elements of a tensor in place with a Fisher-Yates shuffle.
             * @throws std::invalid_argument when t is INT32 or INT64.
             * @see modules/operations/MATH_README.md
             */
            TaskHandle shuffle(Tensor& t);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;

            /// @brief The stream every operation of this view reserves its offset from.
            RNGStream& rng_;
    };

} // namespace SushiBLAS
