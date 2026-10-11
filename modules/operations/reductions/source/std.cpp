/**************************************************************************/
/* std.cpp                                                                */
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
 * @file std.cpp
 * @brief Defines ReductionOps::std, which records the standard deviation reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiBLAS/reductions/centred_reduction.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle ReductionOps::std(const Tensor& t, Tensor& result)
    {
        Dispatch::require_min_elements(t, 1, "math.std", "t");
        Dispatch::require_storage(t, "math.std", "t");

        using DeviationDispatch = Dispatch::CentredReductionDispatch<
            Reduce::CompensatedSum, Dispatch::MeanOfSum,
            Dispatch::SquaredDeviation, Dispatch::RootOfMeanDeviations>;

        // One scratch element holds the mean; the rest are the partials of both stages.
        return Dispatch::execute_reduction(recorder_, t, result, "math.std", "math.std"_op,
                                           DeviationDispatch{},
                                           Dispatch::planned_partials_for(t) + 1);
    }
} // namespace SushiBLAS
