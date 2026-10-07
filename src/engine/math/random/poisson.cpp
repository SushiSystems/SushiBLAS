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

#include <array>
#include <cmath>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle RandomOps::poisson(Tensor& t, double lambda)
    {
        if (!std::isfinite(lambda) || lambda < 0.0 || lambda > K_MAX_POISSON_LAMBDA)
            Dispatch::Detail::throw_operand_error("random.poisson", "lambda",
                                                  "is {}, outside [0, {}]", lambda,
                                                  K_MAX_POISSON_LAMBDA);

        if (t.storage && t.num_elements > 0 && !Device::profile_for(t.get_device()).supports_fp64)
            Dispatch::Detail::throw_operand_error("random.poisson", "t",
                                                  "lives on a device without fp64; random.poisson "
                                                  "draws in double precision for every dtype");

        const Kernels::SYCL::Random::PoissonParams params =
            Kernels::SYCL::Random::make_poisson_params(lambda);

        return Dispatch::execute_random<Dispatch::Domain::REAL>(
            recorder_, rng_, t, "random.poisson", "random.poisson"_op, {lambda},
            [params](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset,
                     int64_t size, auto* pT,
                     const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i)
                {
                    const double draw =
                        Kernels::SYCL::Random::draw_poisson(seed, offset, i, params);
                    return std::array<T, 1>{static_cast<T>(draw)};
                });
            });
    }
} // namespace SushiBLAS
