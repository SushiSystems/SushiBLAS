/**************************************************************************/
/* elu.cpp                                                                */
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

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include "nonlinear_internal.hpp"

namespace SushiBLAS 
{
    sycl::event NonLinearOps::elu(Tensor& t, float alpha) 
    {
        return Internal::execute_nonlinear_forward(engine_, t, "math.nonlinear.elu", "math.nonlinear.elu"_op, 
            [alpha](auto x) { return Internal::greater_than_zero(x) ? x : decltype(x)(alpha) * (Internal::safe_exp(x) - decltype(x)(1)); },
            {alpha});
    }

    sycl::event NonLinearOps::elu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, float alpha)
    {
        return Internal::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.elu_backward", "math.nonlinear.elu_backward"_op, 
            [alpha](auto pdy, auto px) { return pdy * (Internal::greater_than_zero(px) ? decltype(px)(1) : decltype(px)(alpha) * Internal::safe_exp(px)); },
            {alpha});
    }
} // namespace SushiBLAS
