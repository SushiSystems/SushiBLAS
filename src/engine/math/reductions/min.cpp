/**************************************************************************/
/* min.cpp                                                                */
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
 * @file min.cpp
 * @brief Defines ReductionOps::min, which records the minimum reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle ReductionOps::min(const Tensor& t, Tensor& result)
    {
        Dispatch::require_min_elements(t, 1, "math.min", "t");

        return Dispatch::execute_folding_reduction<Reduce::NaNPropagatingMin>(
            recorder_, t, result, "math.min", "math.min"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
