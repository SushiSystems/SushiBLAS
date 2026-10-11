/**************************************************************************/
/* prod.cpp                                                               */
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
 * @file prod.cpp
 * @brief Defines ReductionOps::prod, which records the product reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    // The product folds in a fixed order through RunningProduct; see modules/operations/README.md.
    TaskHandle ReductionOps::prod(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::RunningProduct>(
            recorder_, t, result, "math.prod", "math.prod"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
