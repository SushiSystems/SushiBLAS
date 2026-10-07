/**************************************************************************/
/* argmin.cpp                                                             */
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
 * @file argmin.cpp
 * @brief Defines ReductionOps::argmin, which records the argmin reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include <SushiBLAS/engine/math/reductions/arg_reduction.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle ReductionOps::argmin(const Tensor& t, Tensor& result)
    {
        recorder_.require_sycl_path("Reduction", "math.argmin");
        Dispatch::require_min_elements(t, 1, "math.argmin", "t");
        Dispatch::require_storage(t, "math.argmin", "t");

        return Dispatch::execute_index_reduction(
            recorder_, t, result, "math.argmin", "math.argmin"_op,
            Dispatch::ArgReductionDispatch<Reduce::RunningArgMin>{},
            Dispatch::planned_partials_for(t));
    }
} // namespace SushiBLAS
