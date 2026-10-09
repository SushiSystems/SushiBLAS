/**************************************************************************/
/* logsumexp.cpp                                                          */
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
 * @file logsumexp.cpp
 * @brief Defines ReductionOps::logsumexp, which records the log-sum-exp reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include <SushiBLAS/engine/math/reductions/centred_reduction.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle ReductionOps::logsumexp(const Tensor& t, Tensor& result)
    {
        Dispatch::require_min_elements(t, 1, "math.logsumexp", "t");
        Dispatch::require_storage(t, "math.logsumexp", "t");

        using LogSumExpDispatch = Dispatch::CentredReductionDispatch<
            Reduce::NaNPropagatingMax, Dispatch::ReduceIdentityFinalize,
            Dispatch::ExpBelowCentre, Dispatch::LogPlusCentre>;

        // One scratch element holds the maximum; the rest are the partials of both stages.
        return Dispatch::execute_reduction(recorder_, t, result, "math.logsumexp",
                                           "math.logsumexp"_op, LogSumExpDispatch{},
                                           Dispatch::planned_partials_for(t) + 1);
    }
} // namespace SushiBLAS
