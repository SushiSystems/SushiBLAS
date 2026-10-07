/**************************************************************************/
/* softplus.cpp                                                           */
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
 * @file softplus.cpp
 * @brief Defines NonLinearOps::softplus and NonLinearOps::softplus_backward, which record the
 * softplus activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::softplus(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.softplus", "math.nonlinear.softplus"_op, 
            [](auto x) 
            { 
                using T = decltype(x);
                return T(sycl::fmax(x, T(0)) + sycl::log1p(sycl::exp(-sycl::fabs(x))));
            });
    }

    sycl::event NonLinearOps::softplus_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.softplus_backward", "math.nonlinear.softplus_backward"_op, 
            [](auto pdy, auto px) 
            { 
                return pdy * (decltype(px)(1) - decltype(px)(1) / (decltype(px)(1) + Dispatch::safe_exp(px)));
            });
    }
} // namespace SushiBLAS
