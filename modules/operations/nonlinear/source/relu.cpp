/**************************************************************************/
/* relu.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file relu.cpp
 * @brief Defines NonLinearOps::relu and NonLinearOps::relu_backward, which record the relu
 * activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Literals;

    TaskHandle NonLinearOps::relu(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(recorder_, t, "math.nonlinear.relu", "math.nonlinear.relu"_op, 
            [](auto x) { return Dispatch::greater_than_zero(x) ? x : decltype(x)(0); });
    }

    TaskHandle NonLinearOps::relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(recorder_, dy, x, dx, "math.nonlinear.relu_backward", "math.nonlinear.relu_backward"_op, 
            [](auto pdy, auto px) { return pdy * (Dispatch::greater_than_zero(px) ? decltype(px)(1) : decltype(px)(0)); });
    }
} // namespace SushiBLAS
