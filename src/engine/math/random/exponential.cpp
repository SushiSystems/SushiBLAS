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
        return Dispatch::execute_random(engine_, t, "random.exponential", "random.exponential"_op, {lambda},
            [lambda](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    SB_THROW_IF(true, "Unsupported data type for exponential operation.");
                    return sycl::event();
                }
                else
                {
                    T beta = static_cast<T>(1.0 / lambda);

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        T u = Kernels::SYCL::Random::draw_uniform<T>(seed, offset, i, static_cast<T>(0.0), static_cast<T>(1.0));
                        return -beta * sycl::log(static_cast<T>(1.0) - u);
                    });
                }
            });
    }
} // namespace SushiBLAS
