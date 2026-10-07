/**************************************************************************/
/* he_uniform.cpp                                                         */
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
 * @file he_uniform.cpp
 * @brief Defines RandomOps::he_uniform, which records the He uniform fill of a tensor.
 * @author Mustafa Garip
 */

#include <cmath>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::he_uniform(Tensor& t, int64_t n_in)
    {
        if (n_in <= 0)
            Dispatch::Detail::throw_operand_error("random.he_uniform", "n_in",
                                                  "is {}, expected above 0", n_in);

        const double limit = std::sqrt(6.0 / n_in);

        return Dispatch::execute_random<Dispatch::Domain::REAL>(
            engine_, t, "random.he_uniform", "random.he_uniform"_op,
            {static_cast<double>(n_in), limit},
            [limit](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset,
                    int64_t size, auto* pT,
                    const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                const T lim = static_cast<T>(limit);

                return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t b)
                {
                    return Kernels::SYCL::Random::uniform_block<T>(seed, offset, b, -lim, lim);
                });
            });
    }
} // namespace SushiBLAS
