/**************************************************************************/
/* gelu.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <numbers>
#include <SushiBLAS/ops/math/nonlinear.hpp>
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
