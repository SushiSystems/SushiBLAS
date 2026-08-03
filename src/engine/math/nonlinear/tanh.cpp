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
