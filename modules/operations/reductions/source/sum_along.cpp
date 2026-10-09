/**************************************************************************/
/* sum_along.cpp                                                          */
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
 * @file sum_along.cpp
 * @brief Defines ReductionOps::sum_along, which records the per-axis sum reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The sum is order-sensitive, so it takes the segmented compensated fold; see modules/operations/README.md.
    TaskHandle ReductionOps::sum_along(const Tensor& t, int axis, Tensor& result)
    {
        return Dispatch::execute_folding_axis_reduction<Reduce::CompensatedSum>(
            recorder_, t, axis, result, "math.sum_along", "math.sum_along"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
