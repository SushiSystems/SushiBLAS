/****************************************************************/
/* leaky_relu.cpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file leaky_relu.cpp
 * @brief Defines NonLinearOps::leaky_relu and NonLinearOps::leaky_relu_backward, which record the
 * leaky_relu activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::leaky_relu(Tensor& t, float alpha) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.leaky_relu", "math.nonlinear.leaky_relu"_op, 
            [alpha](auto x) { return Dispatch::greater_than_zero(x) ? x : x * decltype(x)(alpha); },
            {alpha});
    }

    sycl::event NonLinearOps::leaky_relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, float alpha)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.leaky_relu_backward", "math.nonlinear.leaky_relu_backward"_op, 
            [alpha](auto pdy, auto px) { return pdy * (Dispatch::greater_than_zero(px) ? decltype(px)(1) : decltype(px)(alpha)); },
            {alpha});
    }
} // namespace SushiBLAS
