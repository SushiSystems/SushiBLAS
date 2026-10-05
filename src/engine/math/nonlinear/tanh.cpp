/****************************************************************/
/* tanh.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file tanh.cpp
 * @brief Defines NonLinearOps::tanh and NonLinearOps::tanh_backward, which record the tanh
 * activation and its gradient.
 * @author Mustafa Garip
 */

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
