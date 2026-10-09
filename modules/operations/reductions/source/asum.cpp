/**************************************************************************/
/* asum.cpp                                                               */
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
 * @file asum.cpp
 * @brief Defines ReductionOps::asum, which records the sum-of-magnitudes reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // Sum of magnitudes: the same fold as `sum` with `fabs` as the transform.
    TaskHandle ReductionOps::asum(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
            recorder_, t, result, "math.asum", "math.asum"_op,
            [](auto x) { return sycl::fabs(x); },
            Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
