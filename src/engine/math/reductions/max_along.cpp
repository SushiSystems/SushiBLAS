/**************************************************************************/
/* max_along.cpp                                                          */
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
 * @file max_along.cpp
 * @brief Defines ReductionOps::max_along, which records the per-axis maximum reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // An axis reduction has no sycl::reduction form, so it folds; see src/README.md.
    TaskHandle ReductionOps::max_along(const Tensor& t, int axis, Tensor& result)
    {
        Dispatch::require_min_elements(t, 1, "math.max_along", "t");

        return Dispatch::execute_folding_axis_reduction<Reduce::NaNPropagatingMax>(
            recorder_, t, axis, result, "math.max_along", "math.max_along"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
