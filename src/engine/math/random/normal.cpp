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

#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle RandomOps::normal(Tensor& t, double mean, double stddev)
    {
        if (!std::isfinite(mean))
            Dispatch::Detail::throw_operand_error("random.normal", "mean",
                                                  "is {}, expected finite", mean);
        if (!std::isfinite(stddev) || stddev < 0.0)
            Dispatch::Detail::throw_operand_error("random.normal", "stddev",
                                                  "is {}, expected finite and at least 0", stddev);

        return Dispatch::execute_random<Dispatch::Domain::REAL_AND_COMPLEX>(
            recorder_, rng_, t, "random.normal", "random.normal"_op, {mean, stddev},
            [mean, stddev](auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed,
                           uint64_t offset, int64_t size, auto* pT) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    using RealT = typename T::value_type;
                    const RealT m = static_cast<RealT>(mean);
                    const RealT sd =
                        static_cast<RealT>(stddev) / static_cast<RealT>(std::sqrt(2.0));
                    RealT* pR = reinterpret_cast<RealT*>(pT);

                    // A complex fill is one flat run of 2n real draws; see src/README.md.
                    return fill(size * 2, pR, [=](uint64_t b)
                    {
                        return Kernels::SYCL::Random::normal_block<RealT>(seed, offset, b, m, sd);
                    });
                }
                else
                {
                    const T m = static_cast<T>(mean), sd = static_cast<T>(stddev);

                    return fill(size, pT, [=](uint64_t b)
                    {
                        return Kernels::SYCL::Random::normal_block<T>(seed, offset, b, m, sd);
                    });
                }
            });
    }
} // namespace SushiBLAS
