/**************************************************************************/
/* log_normal.cpp                                                         */
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
 * @file log_normal.cpp
 * @brief Defines RandomOps::log_normal, which records the log-normal fill of a tensor.
 * @author Mustafa Garip
 */

#include <cmath>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle RandomOps::log_normal(Tensor& t, double mean, double stddev)
    {
        if (!std::isfinite(mean))
            Dispatch::Detail::throw_operand_error("random.log_normal", "mean",
                                                  "is {}, expected finite", mean);
        if (!std::isfinite(stddev) || stddev < 0.0)
            Dispatch::Detail::throw_operand_error("random.log_normal", "stddev",
                                                  "is {}, expected finite and at least 0", stddev);

        return Dispatch::execute_random<Dispatch::Domain::Real>(
            recorder_, rng_, t, "random.log_normal", "random.log_normal"_op, {mean, stddev},
            [mean, stddev](auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed,
                           uint64_t offset, int64_t size, auto* pT) -> sycl::event
            {
                using T = decltype(scalar_type);
                const T m = static_cast<T>(mean), sd = static_cast<T>(stddev);

                return fill(size, pT, [=](uint64_t b)
                {
                    Kernels::SYCL::Random::DrawBlock<T> values =
                        Kernels::SYCL::Random::normal_block<T>(seed, offset, b, m, sd);
                    for (T& value : values)
                        value = sycl::exp(value);
                    return values;
                });
            });
    }
} // namespace SushiBLAS
