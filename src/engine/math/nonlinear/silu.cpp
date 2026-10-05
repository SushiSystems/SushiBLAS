/****************************************************************/
/* silu.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file silu.cpp
 * @brief Defines NonLinearOps::silu and NonLinearOps::silu_backward, which record the silu
 * activation and its gradient.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::silu(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.silu", "math.nonlinear.silu"_op, 
            [](auto x) { return x / (decltype(x)(1) + Dispatch::safe_exp(-x)); });
    }

    sycl::event NonLinearOps::silu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.silu_backward", "math.nonlinear.silu_backward"_op, 
            [](auto pdy, auto px) { 
                auto sig = decltype(px)(1) / (decltype(px)(1) + Dispatch::safe_exp(-px));
                return pdy * (sig + px * sig * (decltype(px)(1) - sig)); 
            });
    }
} // namespace SushiBLAS
