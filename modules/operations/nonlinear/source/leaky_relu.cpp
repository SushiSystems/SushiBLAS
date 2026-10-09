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

#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle NonLinearOps::leaky_relu(Tensor& t, double alpha) 
    {
        return Dispatch::execute_nonlinear_forward(recorder_, t, "math.nonlinear.leaky_relu", "math.nonlinear.leaky_relu"_op,
            Dispatch::bind_scalars([](auto x, auto a) { return Dispatch::greater_than_zero(x) ? x : x * a; },
                                   Dispatch::ElementScalar(alpha)),
            {alpha});
    }

    TaskHandle NonLinearOps::leaky_relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, double alpha)
    {
        return Dispatch::execute_nonlinear_backward(recorder_, dy, x, dx, "math.nonlinear.leaky_relu_backward", "math.nonlinear.leaky_relu_backward"_op,
            Dispatch::bind_scalars([](auto pdy, auto px, auto a) { return pdy * (Dispatch::greater_than_zero(px) ? decltype(px)(1) : a); },
                                   Dispatch::ElementScalar(alpha)),
            {alpha});
    }
} // namespace SushiBLAS
