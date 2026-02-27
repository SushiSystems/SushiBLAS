/**************************************************************************/
/* leaky_relu.cpp                                                         */
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

#include <SushiBLAS/ops/math/nonlinear.hpp>
#include "nonlinear_internal.hpp"

namespace SushiBLAS 
{
    sycl::event NonLinearOps::leaky_relu(Tensor& t, float alpha) 
    {
        return Internal::execute_nonlinear_forward(engine_, t, "math.nonlinear.leaky_relu", "math.nonlinear.leaky_relu"_op, 
            [alpha](auto x) { return Internal::greater_than_zero(x) ? x : x * decltype(x)(alpha); },
            {alpha});
    }

    sycl::event NonLinearOps::leaky_relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, float alpha)
    {
        return Internal::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.leaky_relu_backward", "math.nonlinear.leaky_relu_backward"_op, 
            [alpha](auto pdy, auto px) { return pdy * (Internal::greater_than_zero(px) ? decltype(px)(1) : decltype(px)(alpha)); },
            {alpha});
    }
} // namespace SushiBLAS
