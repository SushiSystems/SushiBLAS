/**************************************************************************/
/* tanh.cpp                                                               */
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
 * @file tanh.cpp
 * @brief Defines NonLinearOps::tanh and NonLinearOps::tanh_backward, which record the tanh
 * activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::tanh(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.tanh", "math.nonlinear.tanh"_op, 
            [](auto x) { return Dispatch::safe_tanh(x); });
    }

    sycl::event NonLinearOps::tanh_backward(const Tensor& dy, const Tensor& y, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, y, dx, "math.nonlinear.tanh_backward", "math.nonlinear.tanh_backward"_op, 
            [](auto pdy, auto py) 
            {
                return pdy * (decltype(py)(1) - py * py); 
            });
    }
} // namespace SushiBLAS
