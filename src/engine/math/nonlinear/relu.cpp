/****************************************************************/
/* relu.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::relu(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.relu", "math.nonlinear.relu"_op, 
            [](auto x) { return Dispatch::greater_than_zero(x) ? x : decltype(x)(0); });
    }

    sycl::event NonLinearOps::relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.relu_backward", "math.nonlinear.relu_backward"_op, 
            [](auto pdy, auto px) { return pdy * (Dispatch::greater_than_zero(px) ? decltype(px)(1) : decltype(px)(0)); });
    }
} // namespace SushiBLAS
