/**************************************************************************/
/* poisson.hpp                                                            */
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
 * @file poisson.hpp
 * @brief Defines the Poisson draw: Knuth's product below lambda 10, Hoermann's PTRS above.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstdint>
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
                /**
                 * @brief Holds the lambda from which the Poisson draw uses PTRS instead of Knuth's product.
                 * @see kernels/README.md
                 */
                inline constexpr double K_POISSON_PTRS_MIN_LAMBDA = 10.0;

                /**
                 * @brief Holds the Philox blocks reserved per element; it caps the attempts of both methods.
                 * @see kernels/README.md
                 */
                inline constexpr uint64_t K_POISSON_BLOCKS_PER_ELEMENT = 64;

                /**
                 * @brief Holds the per-lambda constants of the Poisson draw, computed once on the host.
                 * @see kernels/README.md
                 */
                struct PoissonParams
                {
                    double lambda = 0.0;        /**< The mean of the distribution. */
                    double exp_neg_lambda = 1.0; /**< exp(-lambda), Knuth's stopping product. */
                    double log_lambda = 0.0;    /**< log(lambda), for the PTRS acceptance test. */
                    double a = 0.0;             /**< PTRS hat parameter a. */
                    double b = 0.0;             /**< PTRS hat parameter b. */
                    double log_inv_alpha = 0.0; /**< log(1 / alpha) of the PTRS hat. */
                    double v_r = 0.0;           /**< PTRS bound below which a draw is accepted at once. */
                };

                /**
                 * @brief Returns the constants of the Poisson draw for @p lambda.
                 * @pre lambda is finite and non-negative.
                 */
                inline PoissonParams make_poisson_params(double lambda)
                {
                    PoissonParams params;
                    params.lambda = lambda;
                    params.exp_neg_lambda = std::exp(-lambda);
                    if (lambda >= K_POISSON_PTRS_MIN_LAMBDA)
                    {
                        const double sqrt_lambda = std::sqrt(lambda);
                        params.log_lambda = std::log(lambda);
                        params.b = 0.931 + 2.53 * sqrt_lambda;
                        params.a = -0.059 + 0.02483 * params.b;
                        params.log_inv_alpha = std::log(1.1239 + 1.1328 / (params.b - 3.4));
                        params.v_r = 0.9277 - 3.6224 / (params.b - 2.0);
                    }
                    return params;
                }

                /**
                 * @brief Draws one Poisson count by multiplying uniforms until the product reaches exp(-lambda).
                 * @pre lambda < K_POISSON_PTRS_MIN_LAMBDA.
                 */
                inline double poisson_knuth(uint64_t seed, uint64_t offset, uint64_t element,
                                            const PoissonParams& params)
                {
                    const uint64_t first_block = element * K_POISSON_BLOCKS_PER_ELEMENT;
                    double product = 1.0;
                    double count = 0.0;
                    for (uint64_t block = 0; block < K_POISSON_BLOCKS_PER_ELEMENT; ++block)
                    {
                        const DrawBlock<double> u = uniform01_block<double>(
                            philox_block(seed, offset, first_block + block));
                        for (double draw : u)
                        {
                            product *= draw;
                            if (product <= params.exp_neg_lambda)
                                return count;
                            count += 1.0;
                        }
                    }
                    return count;
                }

                /**
                 * @brief Draws one Poisson count by Hoermann's transformed rejection with squeeze (PTRS).
                 * @pre lambda >= K_POISSON_PTRS_MIN_LAMBDA.
                 * @see kernels/README.md
                 */
                inline double poisson_ptrs(uint64_t seed, uint64_t offset, uint64_t element,
                                           const PoissonParams& params)
                {
                    const uint64_t first_block = element * K_POISSON_BLOCKS_PER_ELEMENT;
                    for (uint64_t attempt = 0; attempt < K_POISSON_BLOCKS_PER_ELEMENT; ++attempt)
                    {
                        const DrawBlock<double> uv = uniform01_block<double>(
                            philox_block(seed, offset, first_block + attempt));
                        const double u = uv[0] - 0.5;
                        const double v = uv[1];
                        const double us = 0.5 - sycl::fabs(u);
                        const double k = sycl::floor((2.0 * params.a / us + params.b) * u + params.lambda + 0.43);

                        if (us >= 0.07 && v <= params.v_r)
                            return k;
                        // A zero us makes k infinite or NaN; both fail this test and are redrawn.
                        if (!(k >= 0.0) || (us < 0.013 && v > us))
                            continue;

                        const double log_hat = sycl::log(v) + params.log_inv_alpha
                                               - sycl::log(params.a / (us * us) + params.b);
                        const double log_target = -params.lambda + k * params.log_lambda - sycl::lgamma(k + 1.0);
                        if (log_hat <= log_target)
                            return k;
                    }
                    return sycl::floor(params.lambda);
                }

                /**
                 * @brief Draws the Poisson count of element @p element, choosing the method by lambda.
                 * @see kernels/README.md
                 */
                inline double draw_poisson(uint64_t seed, uint64_t offset, uint64_t element,
                                           const PoissonParams& params)
                {
                    if (params.lambda < K_POISSON_PTRS_MIN_LAMBDA)
                        return poisson_knuth(seed, offset, element, params);
                    return poisson_ptrs(seed, offset, element, params);
                }

            } // namespace Random
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
