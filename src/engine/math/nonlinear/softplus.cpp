/****************************************************************/
/* softplus.cpp                                                 */
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

    sycl::event NonLinearOps::softplus(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.softplus", "math.nonlinear.softplus"_op, 
            [](auto x) 
            { 
                return Dispatch::safe_log(decltype(x)(1) + Dispatch::safe_exp(x));
            });
    }

    sycl::event NonLinearOps::softplus_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.softplus_backward", "math.nonlinear.softplus_backward"_op, 
            [](auto pdy, auto px) 
            { 
                return pdy * (decltype(px)(1) - decltype(px)(1) / (decltype(px)(1) + Dispatch::safe_exp(px)));
            });
    }
} // namespace SushiBLAS
