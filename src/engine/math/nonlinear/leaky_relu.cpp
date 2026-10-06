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

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::leaky_relu(Tensor& t, float alpha) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.leaky_relu", "math.nonlinear.leaky_relu"_op, 
            [alpha](auto x) { return Dispatch::greater_than_zero(x) ? x : x * decltype(x)(alpha); },
            {alpha});
    }

    sycl::event NonLinearOps::leaky_relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, float alpha)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.leaky_relu_backward", "math.nonlinear.leaky_relu_backward"_op, 
            [alpha](auto pdy, auto px) { return pdy * (Dispatch::greater_than_zero(px) ? decltype(px)(1) : decltype(px)(alpha)); },
            {alpha});
    }
} // namespace SushiBLAS
