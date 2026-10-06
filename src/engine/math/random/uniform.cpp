/**************************************************************************/
/* uniform.cpp                                                            */
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
 * @file uniform.cpp
 * @brief Defines RandomOps::uniform, which records the uniform fill of a tensor.
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

    sycl::event RandomOps::uniform(Tensor& t, double min, double max)
    {
        return Dispatch::execute_random(engine_, t, "random.uniform", "random.uniform"_op, {min, max},
            [min, max](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if constexpr (Core::is_complex_v<T>)
                {
                    using RealT = typename T::value_type;
                    RealT lo = static_cast<RealT>(min), hi = static_cast<RealT>(max);
                    RealT* pR = reinterpret_cast<RealT*>(pT);
                    int64_t n = size * 2;

                    // A complex fill is one flat run of 2n real draws; see src/README.md.
                    return Dispatch::submit_random_fill(q, deps, n, pR, [=](uint64_t i) -> RealT
                    {
                        return Kernels::SYCL::Random::draw_uniform<RealT>(seed, offset, i, lo, hi);
                    });
                }
                else
                {
                    T lo = static_cast<T>(min), hi = static_cast<T>(max);

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        return Kernels::SYCL::Random::draw_uniform<T>(seed, offset, i, lo, hi);
                    });
                }
            });
    }
} // namespace SushiBLAS
