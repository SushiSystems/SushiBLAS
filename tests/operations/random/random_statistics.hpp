/**************************************************************************/
/* random_statistics.hpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file random_statistics.hpp
 * @brief Declares the laws of the random draws and the checks that a sample matches one within standard errors.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds how many standard errors a sample moment may lie from the law's moment; at 5 the false-failure rate of one moment on a fixed seed is about 6e-7.
        inline constexpr double K_STANDARD_ERRORS = 5.0;

        /** @brief Holds the moments of a distribution that fix the standard errors of a sample. */
        struct Law
        {
            double mean;           /**< The expected value. */
            double variance;       /**< The second central moment. */
            double fourth_central; /**< The fourth central moment. */
        };

        /** @brief Holds what a sample shows of its law. */
        struct SampleMoments
        {
            double mean;     /**< The sample mean. */
            double variance; /**< The unbiased sample variance. */
            double lowest;   /**< The smallest element. */
            double highest;  /**< The largest element. */
        };

        /** @brief Returns E[(X - m1)^4] from the raw moments m1 to m4. */
        [[nodiscard]] inline double fourth_central_of_raw(double m1, double m2, double m3, double m4) noexcept
        {
            return m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 * m1 - 3.0 * m1 * m1 * m1 * m1;
        }

        /** @brief Returns the law of N(mean, stddev^2). */
        [[nodiscard]] inline Law normal_law(double mean, double stddev) noexcept
        {
            const double variance = stddev * stddev;
            return Law{mean, variance, 3.0 * variance * variance};
        }

        /** @brief Returns the law of the continuous uniform draw on [min, max). */
        [[nodiscard]] inline Law uniform_law(double min, double max) noexcept
        {
            const double width = max - min;
            return Law{(min + max) / 2.0, width * width / 12.0, std::pow(width, 4.0) / 80.0};
        }

        /** @brief Returns the law of the exponential draw with the given rate. */
        [[nodiscard]] inline Law exponential_law(double rate) noexcept
        {
            return Law{1.0 / rate, 1.0 / (rate * rate), 9.0 / std::pow(rate, 4.0)};
        }

        /** @brief Returns the law of exp(N(mean, stddev^2)), whose raw moments are exp(k mean + k^2 stddev^2 / 2). */
        [[nodiscard]] inline Law log_normal_law(double mean, double stddev) noexcept
        {
            const auto raw = [mean, stddev](double k)
            {
                return std::exp(k * mean + k * k * stddev * stddev / 2.0);
            };
            const double m1 = raw(1.0);
            const double m2 = raw(2.0);
            return Law{m1, m2 - m1 * m1, fourth_central_of_raw(m1, m2, raw(3.0), raw(4.0))};
        }

        /** @brief Returns the law of the Bernoulli draw with probability p of a 1. */
        [[nodiscard]] inline Law bernoulli_law(double p) noexcept
        {
            const double q = 1.0 - p;
            return Law{p, p * q, p * q * (q * q * q + p * p * p)};
        }

        /** @brief Returns the law of the integer uniform draw on low to high, both included. */
        [[nodiscard]] inline Law discrete_uniform_law(double low, double high) noexcept
        {
            const double n = high - low + 1.0;
            return Law{(low + high) / 2.0, (n * n - 1.0) / 12.0,
                       (n * n - 1.0) * (3.0 * n * n - 7.0) / 240.0};
        }

        /** @brief Returns the law of the Poisson draw with mean lambda, whose fourth central moment is lambda + 3 lambda^2. */
        [[nodiscard]] inline Law poisson_law(double lambda) noexcept
        {
            return Law{lambda, lambda, lambda + 3.0 * lambda * lambda};
        }

        /**
         * @brief Returns the mean and the variance of N(0, 1) cut to [lower, upper].
         * @note The fourth central moment is left zero; only the mean is checked against this law.
         */
        [[nodiscard]] inline Law truncated_normal_law(double lower, double upper) noexcept
        {
            const double inv_sqrt_two_pi = 0.3989422804014327;
            const auto pdf = [inv_sqrt_two_pi](double x)
            {
                return inv_sqrt_two_pi * std::exp(-x * x / 2.0);
            };
            const auto cdf = [](double x)
            {
                return 0.5 * std::erfc(-x / std::sqrt(2.0));
            };
            const auto pdf_times = [&pdf](double x)
            {
                return pdf(x) == 0.0 ? 0.0 : x * pdf(x);
            };

            const double mass = cdf(upper) - cdf(lower);
            const double mean = (pdf(lower) - pdf(upper)) / mass;
            const double variance = 1.0 + (pdf_times(lower) - pdf_times(upper)) / mass - mean * mean;
            return Law{mean, variance, 0.0};
        }

        /** @brief Returns the mean, the unbiased variance and the extremes of the samples. */
        [[nodiscard]] inline SampleMoments measure_moments(const std::vector<double>& samples)
        {
            double sum = 0.0;
            for (const double value : samples)
                sum += value;
            const double mean = sum / static_cast<double>(samples.size());

            double squares = 0.0;
            for (const double value : samples)
                squares += (value - mean) * (value - mean);

            const auto extremes = std::minmax_element(samples.begin(), samples.end());
            return SampleMoments{mean, squares / static_cast<double>(samples.size() - 1),
                                 *extremes.first, *extremes.second};
        }

        /** @brief Returns K_STANDARD_ERRORS standard errors of a sample mean of n draws from the law. */
        [[nodiscard]] inline double mean_bound(const Law& law, std::size_t n) noexcept
        {
            return K_STANDARD_ERRORS * std::sqrt(law.variance / static_cast<double>(n));
        }

        /** @brief Returns K_STANDARD_ERRORS standard errors of a sample variance of n draws from the law. */
        [[nodiscard]] inline double variance_bound(const Law& law, std::size_t n) noexcept
        {
            return K_STANDARD_ERRORS
                 * std::sqrt((law.fourth_central - law.variance * law.variance)
                             / static_cast<double>(n));
        }

        /** @brief Expects the sample mean to lie within mean_bound of the law's mean. */
        inline void expect_mean_matches(const std::vector<double>& samples, const Law& law)
        {
            EXPECT_NEAR(measure_moments(samples).mean, law.mean, mean_bound(law, samples.size()));
        }

        /** @brief Expects the sample mean and variance to lie within their standard errors of the law's. */
        inline void expect_moments_match(const std::vector<double>& samples, const Law& law)
        {
            const SampleMoments measured = measure_moments(samples);
            EXPECT_NEAR(measured.mean, law.mean, mean_bound(law, samples.size()));
            EXPECT_NEAR(measured.variance, law.variance, variance_bound(law, samples.size()));
        }

        /** @brief Widens the elements of a host vector to double. */
        template <typename T>
        [[nodiscard]] std::vector<double> widened(const std::vector<T>& values)
        {
            return std::vector<double>(values.begin(), values.end());
        }
    } // namespace Test
} // namespace SushiBLAS
