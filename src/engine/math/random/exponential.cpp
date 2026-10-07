/**************************************************************************/
/* exponential.cpp                                                        */
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
 * @file exponential.cpp
 * @brief Defines RandomOps::exponential, which records the exponential fill of a tensor.
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

    sycl::event RandomOps::exponential(Tensor& t, double lambda)
    {
        if (!std::isfinite(lambda) || !(lambda > 0.0))
            Dispatch::Detail::throw_operand_error("random.exponential", "lambda",
                                                  "is {}, expected finite and above 0", lambda);

        return Dispatch::execute_random<Dispatch::Domain::REAL>(
            engine_, t, "random.exponential", "random.exponential"_op, {lambda},
            [lambda](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset,
                     int64_t size, auto* pT,
                     const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                const T beta = static_cast<T>(1.0 / lambda);

                return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t b)
                {
                    Kernels::SYCL::Random::DrawBlock<T> values =
                        Kernels::SYCL::Random::uniform_block<T>(seed, offset, b, static_cast<T>(0.0),
                                                                static_cast<T>(1.0));
                    for (T& value : values)
                        value = -beta * sycl::log1p(-value);
                    return values;
                });
            });
    }
} // namespace SushiBLAS
