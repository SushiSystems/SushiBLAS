/**************************************************************************/
/* discrete_uniform.cpp                                                   */
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
 * @file discrete_uniform.cpp
 * @brief Defines RandomOps::discrete_uniform, which records the discrete uniform fill of a tensor.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Holds 2^24, the largest magnitude below which FLOAT32 holds every integer. */
        constexpr int64_t K_MAX_EXACT_FLOAT32_INTEGER = int64_t{1} << 24;
    } // namespace

    TaskHandle RandomOps::discrete_uniform(Tensor& t, int32_t min, int32_t max)
    {
        if (min > max)
            Dispatch::Detail::throw_operand_error("random.discrete_uniform", "min",
                                                  "is {}, greater than max {}", min, max);

        const int64_t magnitude = std::max(std::abs(static_cast<int64_t>(min)),
                                           std::abs(static_cast<int64_t>(max)));
        if (t.dtype == Core::DataType::Float32 && magnitude > K_MAX_EXACT_FLOAT32_INTEGER)
            Dispatch::Detail::throw_operand_error("random.discrete_uniform", "t",
                                                  "is FLOAT32, which cannot hold every integer in "
                                                  "[{}, {}]; the bound is {} in magnitude",
                                                  min, max, K_MAX_EXACT_FLOAT32_INTEGER);

        return Dispatch::execute_random<Dispatch::Domain::Real>(recorder_, rng_, t, "random.discrete_uniform", "random.discrete_uniform"_op, {static_cast<double>(min), static_cast<double>(max)},
            [min, max](auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed,
                       uint64_t offset, int64_t size, auto* pT) -> sycl::event
            {
                using T = decltype(scalar_type);
                const int64_t lo = static_cast<int64_t>(min);
                const uint64_t count = static_cast<uint64_t>(static_cast<int64_t>(max) - lo + 1);

                return fill(size, pT, [=](uint64_t i)
                {
                    const uint64_t pick = Kernels::SYCL::Random::draw_index(seed, offset, i, count);
                    return std::array<T, 1>{static_cast<T>(lo + static_cast<int64_t>(pick))};
                });
            });
    }
} // namespace SushiBLAS
