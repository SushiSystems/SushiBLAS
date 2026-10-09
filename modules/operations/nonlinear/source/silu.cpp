/**************************************************************************/
/* silu.cpp                                                               */
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
 * @file silu.cpp
 * @brief Defines NonLinearOps::silu and NonLinearOps::silu_backward, which record the silu
 * activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle NonLinearOps::silu(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(recorder_, t, "math.nonlinear.silu", "math.nonlinear.silu"_op, 
            [](auto x) { return x / (decltype(x)(1) + Dispatch::safe_exp(-x)); });
    }

    TaskHandle NonLinearOps::silu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(recorder_, dy, x, dx, "math.nonlinear.silu_backward", "math.nonlinear.silu_backward"_op, 
            [](auto pdy, auto px) { 
                auto sig = decltype(px)(1) / (decltype(px)(1) + Dispatch::safe_exp(-px));
                return pdy * (sig + px * sig * (decltype(px)(1) - sig)); 
            });
    }
} // namespace SushiBLAS
