/****************************************************************/
/* random.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
             * @brief Fill tensor with a constant value.
             * @param t Tensor to fill.
             * @param value The constant value.
             * @return sycl::event.
             */
            sycl::event constant(Tensor& t, double value);

            /**
             * @brief Fill tensor with values from a uniform distribution [min, max).
             * @param t Tensor to fill.
             * @param min Minimum value (default 0.0).
             * @param max Maximum value (default 1.0).
             * @return sycl::event.
             */
            sycl::event uniform(Tensor& t, double min = 0.0, double max = 1.0);

            /**
             * @brief Fill tensor with values from a normal (Gaussian) distribution.
             * @param t Tensor to fill.
             * @param mean Mean of the distribution (default 0.0).
             * @param stddev Standard deviation (default 1.0).
             * @return sycl::event.
             */
            sycl::event normal(Tensor& t, double mean = 0.0, double stddev = 1.0);

            /**
             * @brief Fill tensor with values from a Log-Normal distribution.
             * @param t Tensor to fill.
             * @param mean Mean of the underlying normal distribution.
             * @param stddev Standard deviation of the underlying normal distribution.
             * @return sycl::event.
             */
            sycl::event log_normal(Tensor& t, double mean = 0.0, double stddev = 1.0);

            /**
             * @brief Fill tensor with values from an Exponential distribution.
             * @param t Tensor to fill.
             * @param lambda Rate parameter (1/mean).
             * @return sycl::event.
             */
            sycl::event exponential(Tensor& t, double lambda = 1.0);

            /**
             * @brief Fill tensor with values from a Poisson distribution.
             * @param t Tensor to fill (should be an integer type or float).
             * @param lambda Mean of the distribution.
             * @return sycl::event.
             */
            sycl::event poisson(Tensor& t, double lambda = 1.0);

            /**
             * @brief Fill tensor with values from a Discrete Uniform distribution [min, max).
             * @param t Tensor to fill.
             * @param min Minimum value (inclusive).
             * @param max Maximum value (exclusive).
             * @return sycl::event.
             */
            sycl::event discrete_uniform(Tensor& t, int32_t min, int32_t max);

            /**
             * @brief Fills a tensor from a truncated normal distribution.
             * @param a Lower bound, in standard deviations.
             * @param b Upper bound, in standard deviations.
             */
            sycl::event truncated_normal(Tensor& t, double mean = 0.0, double stddev = 1.0, double a = -2.0, double b = 2.0);

            /**
             * @brief Fill tensor with values from a Bernoulli distribution.
             * @param t Tensor to fill.
             * @param p Probability of success (1.0).
             * @return sycl::event.
             */
            sycl::event bernoulli(Tensor& t, double p = 0.5);

            /**
             * @brief Xavier/Glorot Uniform initialization.
             * @param t Tensor to initialize.
             * @param n_in Number of input features.
             * @param n_out Number of output features.
             * @return sycl::event.
             */
            sycl::event xavier_uniform(Tensor& t, int64_t n_in, int64_t n_out);

            /**
             * @brief Xavier/Glorot Normal initialization.
             * @param t Tensor to initialize.
             * @param n_in Number of input features.
             * @param n_out Number of output features.
             * @return sycl::event.
             */
            sycl::event xavier_normal(Tensor& t, int64_t n_in, int64_t n_out);

            /**
             * @brief He (Kaiming) Uniform initialization.
             * @param t Tensor to initialize.
             * @param n_in Number of input features.
             * @return sycl::event.
             */
            sycl::event he_uniform(Tensor& t, int64_t n_in);

            /**
             * @brief He (Kaiming) Normal initialization.
             * @param t Tensor to initialize.
             * @param n_in Number of input features.
             * @return sycl::event.
             */
            sycl::event he_normal(Tensor& t, int64_t n_in);

            /**
             * @brief Fills a matrix with a random (semi-)orthogonal matrix scaled by gain.
             * @param t Must be rank 2, non-empty, and FLOAT32 or FLOAT64.
             * @throws std::runtime_error if the tensor is not a non-empty real-valued matrix.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event orthogonal(Tensor& t, double gain = 1.0);

            /**
             * @brief Permutes the elements of a tensor in place with a Fisher-Yates shuffle.
             * @see include/SushiBLAS/engine/math/README.md
             */
            sycl::event shuffle(Tensor& t);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
