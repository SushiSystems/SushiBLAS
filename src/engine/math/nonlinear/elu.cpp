/****************************************************************/
/* elu.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file elu.cpp
 * @brief Defines NonLinearOps::elu and NonLinearOps::elu_backward, which record the elu activation
 * and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::elu(Tensor& t, float alpha) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.elu", "math.nonlinear.elu"_op, 
            [alpha](auto x) { return Dispatch::greater_than_zero(x) ? x : decltype(x)(alpha) * (Dispatch::safe_exp(x) - decltype(x)(1)); },
            {alpha});
    }

    sycl::event NonLinearOps::elu_backward(const Tensor& dy, const Tensor& x, Tensor& dx, float alpha)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.elu_backward", "math.nonlinear.elu_backward"_op, 
            [alpha](auto pdy, auto px) { return pdy * (Dispatch::greater_than_zero(px) ? decltype(px)(1) : decltype(px)(alpha) * Dispatch::safe_exp(px)); },
            {alpha});
    }
} // namespace SushiBLAS
