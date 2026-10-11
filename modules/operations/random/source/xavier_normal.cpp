/**************************************************************************/
/* xavier_normal.cpp                                                      */
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
 * @file xavier_normal.cpp
 * @brief Defines RandomOps::xavier_normal, which records the Xavier normal fill of a tensor.
 * @author Mustafa Garip
 */

#include <cmath>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle RandomOps::xavier_normal(Tensor& t, int64_t n_in, int64_t n_out)
    {
        if (n_in + n_out <= 0)
            Dispatch::Detail::throw_operand_error("random.xavier_normal", "n_in + n_out",
                                                  "is {}, expected above 0", n_in + n_out);

        const double stddev = std::sqrt(2.0 / (n_in + n_out));

        return Dispatch::execute_random<Dispatch::Domain::Real>(
            recorder_, rng_, t, "random.xavier_normal", "random.xavier_normal"_op,
            {static_cast<double>(n_in), static_cast<double>(n_out), stddev},
            [stddev](auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed,
                     uint64_t offset, int64_t size, auto* pT) -> sycl::event
            {
                using T = decltype(scalar_type);
                const T sd = static_cast<T>(stddev);

                return fill(size, pT, [=](uint64_t b)
                {
                    return Kernels::SYCL::Random::normal_block<T>(seed, offset, b,
                                                                  static_cast<T>(0.0), sd);
                });
            });
    }
} // namespace SushiBLAS
