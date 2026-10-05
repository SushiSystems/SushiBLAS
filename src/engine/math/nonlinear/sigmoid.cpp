/****************************************************************/
/* sigmoid.cpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file sigmoid.cpp
 * @brief Defines NonLinearOps::sigmoid and NonLinearOps::sigmoid_backward, which record the
 * sigmoid activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::sigmoid(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.sigmoid", "math.nonlinear.sigmoid"_op, 
            [](auto x) { return decltype(x)(1) / (decltype(x)(1) + Dispatch::safe_exp(-x)); });
    }

    sycl::event NonLinearOps::sigmoid_backward(const Tensor& dy, const Tensor& y, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, y, dx, "math.nonlinear.sigmoid_backward", "math.nonlinear.sigmoid_backward"_op, 
            [](auto pdy, auto py) { 
                return pdy * py * (decltype(py)(1) - py); 
            });
    }
} // namespace SushiBLAS
