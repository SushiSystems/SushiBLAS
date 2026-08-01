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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <numbers>
#include <SushiBLAS/engine/math/nonlinear.hpp>
#include "nonlinear_internal.hpp"

namespace SushiBLAS 
{
    sycl::event NonLinearOps::gelu(Tensor& t) 
    {
        return Internal::execute_nonlinear_forward(engine_, t, "math.nonlinear.gelu", "math.nonlinear.gelu"_op, 
            [](auto x) { 
                auto x3 = x * x * x;
                auto inner = decltype(x)(0.7978845608028654) * (x + decltype(x)(0.044715) * x3);
                return decltype(x)(0.5) * x * (decltype(x)(1) + Internal::safe_tanh(inner)); 
            });
    }

    sycl::event NonLinearOps::gelu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Internal::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.gelu_backward", "math.nonlinear.gelu_backward"_op, 
            [](auto pdy, auto px) { 
                auto x3 = px * px * px;
                auto inner = decltype(px)(0.7978845608028654) * (px + decltype(px)(0.044715) * x3);
                auto t = Internal::safe_tanh(inner);
                auto cosh_term = decltype(px)(1) - t * t;
                auto d_inner = decltype(px)(0.7978845608028654) * (decltype(px)(1) + decltype(px)(0.134145) * px * px);
                auto pdf = decltype(px)(0.5) * px * cosh_term * d_inner;
                auto cdf = decltype(px)(0.5) * (decltype(px)(1) + t);
                return pdy * (cdf + pdf); 
            });
    }
} // namespace SushiBLAS
