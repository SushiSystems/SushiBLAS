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

#include <cmath>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/task/task_types.hpp>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Reports whether [min, max] is a finite, ordered interval of finite width in R. */
        template<typename R>
        bool is_drawable_interval(double min, double max)
        {
            const R lo = static_cast<R>(min);
            const R hi = static_cast<R>(max);
            return std::isfinite(lo) && std::isfinite(hi) && lo <= hi && std::isfinite(hi - lo);
        }
    } // namespace

    TaskHandle RandomOps::uniform(Tensor& t, double min, double max)
    {
        const bool single = t.dtype == Core::DataType::Float32 || t.dtype == Core::DataType::Complex32;
        const bool drawable = std::isfinite(min) && std::isfinite(max)
                              && (single ? is_drawable_interval<float>(min, max)
                                         : is_drawable_interval<double>(min, max));
        if (!drawable)
            Dispatch::Detail::throw_operand_error("random.uniform", "min",
                                                  "and max are {} and {}; uniform needs finite "
                                                  "bounds with min <= max and a finite width in {}",
                                                  min, max, Core::to_string(t.dtype));

        return Dispatch::execute_random<Dispatch::Domain::RealAndComplex>(recorder_, rng_, t, "random.uniform", "random.uniform"_op, {min, max},
            [min, max](auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed,
                       uint64_t offset, int64_t size, auto* pT) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    using RealT = typename T::value_type;
                    const RealT lo = static_cast<RealT>(min), hi = static_cast<RealT>(max);
                    RealT* pR = reinterpret_cast<RealT*>(pT);

                    // A complex fill is one flat run of 2n real draws; see modules/operations/README.md.
                    return fill(size * 2, pR, [=](uint64_t b)
                    {
                        return Kernels::SYCL::Random::uniform_block<RealT>(seed, offset, b, lo, hi);
                    });
                }
                else
                {
                    const T lo = static_cast<T>(min), hi = static_cast<T>(max);

                    return fill(size, pT, [=](uint64_t b)
                    {
                        return Kernels::SYCL::Random::uniform_block<T>(seed, offset, b, lo, hi);
                    });
                }
            });
    }
} // namespace SushiBLAS
