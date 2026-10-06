/**************************************************************************/
/* poisson.cpp                                                            */
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
 * @file poisson.cpp
 * @brief Defines RandomOps::poisson, which records the Poisson fill of a tensor.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::poisson(Tensor& t, double lambda)
    {
        return Dispatch::execute_random(engine_, t, "random.poisson", "random.poisson"_op, {lambda},
            [lambda](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    SB_THROW_IF(true, "Unsupported data type for poisson operation.");
                    return sycl::event();
                }
                else
                {
                    // Knuth's algorithm, capped at MAX_ITERS draws per element; see src/README.md.
                    double L = sycl::exp(-lambda);
                    constexpr int MAX_ITERS = 10000;

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        int32_t k = 0;
                        double running_p = 1.0;
                        for (int attempt = 0; attempt < MAX_ITERS; ++attempt)
                        {
                            uint64_t draw_index = i * static_cast<uint64_t>(MAX_ITERS) + static_cast<uint64_t>(attempt);
                            double u = Kernels::SYCL::Random::draw_uniform<double>(seed, offset, draw_index, 0.0, 1.0);
                            running_p *= u;
                            if (running_p <= L) break;
                            ++k;
                        }
                        return static_cast<T>(k);
                    });
                }
            });
    }
} // namespace SushiBLAS
