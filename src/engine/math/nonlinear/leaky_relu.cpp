/**************************************************************************/
/* leaky_relu.cpp                                                         */
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
 * @file leaky_relu.cpp
 * @brief Defines NonLinearOps::leaky_relu and NonLinearOps::leaky_relu_backward, which record the
 * leaky_relu activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>
#include <SushiBLAS/engine/math/elementwise/scalar.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle NonLinearOps::leaky_relu(Tensor& t, double alpha) 
    {
        const Dispatch::ElementScalar a(alpha);
        return Dispatch::execute_nonlinear_forward(recorder_, t, "math.nonlinear.leaky_relu", "math.nonlinear.leaky_relu"_op, 
            [a](auto x) { return Dispatch::greater_than_zero(x) ? x : x * a.as<decltype(x)>(); },
            {alpha});
    }

    TaskHandle NonLinearOps::leaky_relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, double alpha)
    {
        const Dispatch::ElementScalar a(alpha);
        return Dispatch::execute_nonlinear_backward(recorder_, dy, x, dx, "math.nonlinear.leaky_relu_backward", "math.nonlinear.leaky_relu_backward"_op, 
            [a](auto pdy, auto px) { return pdy * (Dispatch::greater_than_zero(px) ? decltype(px)(1) : a.as<decltype(px)>()); },
            {alpha});
    }
} // namespace SushiBLAS
