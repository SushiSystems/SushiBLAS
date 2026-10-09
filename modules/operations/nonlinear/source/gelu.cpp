/**************************************************************************/
/* gelu.cpp                                                               */
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
 * @file gelu.cpp
 * @brief Defines NonLinearOps::gelu and NonLinearOps::gelu_backward, which record the gelu
 * activation and its gradient.
 * @author Mustafa Garip
 */

#include <numbers>
#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle NonLinearOps::gelu(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(recorder_, t, "math.nonlinear.gelu", "math.nonlinear.gelu"_op, 
            [](auto x) { 
                auto x3 = x * x * x;
                auto inner = decltype(x)(0.7978845608028654) * (x + decltype(x)(0.044715) * x3);
                return decltype(x)(0.5) * x * (decltype(x)(1) + Dispatch::safe_tanh(inner)); 
            });
    }

    TaskHandle NonLinearOps::gelu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(recorder_, dy, x, dx, "math.nonlinear.gelu_backward", "math.nonlinear.gelu_backward"_op, 
            [](auto pdy, auto px) { 
                auto x3 = px * px * px;
                auto inner = decltype(px)(0.7978845608028654) * (px + decltype(px)(0.044715) * x3);
                auto t = Dispatch::safe_tanh(inner);
                auto cosh_term = decltype(px)(1) - t * t;
                auto d_inner = decltype(px)(0.7978845608028654) * (decltype(px)(1) + decltype(px)(0.134145) * px * px);
                auto pdf = decltype(px)(0.5) * px * cosh_term * d_inner;
                auto cdf = decltype(px)(0.5) * (decltype(px)(1) + t);
                return pdy * (cdf + pdf); 
            });
    }
} // namespace SushiBLAS
