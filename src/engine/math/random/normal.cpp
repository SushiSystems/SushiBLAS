/**************************************************************************/
/* normal.cpp                                                             */
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
 * @file normal.cpp
 * @brief Defines RandomOps::normal, which records the normal fill of a tensor.
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

    sycl::event RandomOps::normal(Tensor& t, double mean, double stddev)
    {
        return Dispatch::execute_random(engine_, t, "random.normal", "random.normal"_op, {mean, stddev},
            [mean, stddev](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if constexpr (Core::is_complex_v<T>)
                {
                    using RealT = typename T::value_type;
                    RealT m = static_cast<RealT>(mean);
                    RealT sd = static_cast<RealT>(stddev) / static_cast<RealT>(std::sqrt(2.0));
                    RealT* pR = reinterpret_cast<RealT*>(pT);
                    int64_t n = size * 2;

                    // A complex fill is one flat run of 2n real draws; see src/README.md.
                    return Dispatch::submit_random_fill(q, deps, n, pR, [=](uint64_t i) -> RealT
                    {
                        return Kernels::SYCL::Random::draw_normal<RealT>(seed, offset, i, m, sd);
                    });
                }
                else
                {
                    T m = static_cast<T>(mean), sd = static_cast<T>(stddev);

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        return Kernels::SYCL::Random::draw_normal<T>(seed, offset, i, m, sd);
                    });
                }
            });
    }
} // namespace SushiBLAS
