/**************************************************************************/
/* bernoulli.cpp                                                          */
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
 * @file bernoulli.cpp
 * @brief Defines RandomOps::bernoulli, which records the Bernoulli fill of a tensor.
 * @author Mustafa Garip
 */

#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/task/task_types.hpp>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle RandomOps::bernoulli(Tensor& t, double p)
    {
        if (!(p >= 0.0 && p <= 1.0))
            Dispatch::Detail::throw_operand_error("random.bernoulli", "p",
                                                  "is {}, expected within [0, 1]", p);

        return Dispatch::execute_random<Dispatch::Domain::Real>(
            recorder_, rng_, t, "random.bernoulli", "random.bernoulli"_op, {p},
            [p](auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed, uint64_t offset,
                int64_t size, auto* pT) -> sycl::event
            {
                using T = decltype(scalar_type);
                const T pt = static_cast<T>(p);

                return fill(size, pT, [=](uint64_t b)
                {
                    Kernels::SYCL::Random::DrawBlock<T> values =
                        Kernels::SYCL::Random::uniform_block<T>(seed, offset, b, static_cast<T>(0.0),
                                                                static_cast<T>(1.0));
                    for (T& value : values)
                        value = (value < pt) ? static_cast<T>(1.0) : static_cast<T>(0.0);
                    return values;
                });
            });
    }
} // namespace SushiBLAS
