/****************************************************************/
/* gelu.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <numbers>
#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/nonlinear/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event NonLinearOps::gelu(Tensor& t) 
    {
        return Dispatch::execute_nonlinear_forward(engine_, t, "math.nonlinear.gelu", "math.nonlinear.gelu"_op, 
            [](auto x) { 
                auto x3 = x * x * x;
                auto inner = decltype(x)(0.7978845608028654) * (x + decltype(x)(0.044715) * x3);
                return decltype(x)(0.5) * x * (decltype(x)(1) + Dispatch::safe_tanh(inner)); 
            });
    }

    sycl::event NonLinearOps::gelu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        return Dispatch::execute_nonlinear_backward(engine_, dy, x, dx, "math.nonlinear.gelu_backward", "math.nonlinear.gelu_backward"_op, 
            [](auto pdy, auto px) { 
                auto x3 = px * px * px;
                auto inner = decltype(px)(0.7978845608028654) * (px + decltype(px)(0.044715) * x3);
                auto t = Dispatch::safe_tanh(inner);
                auto cosh_term = decltype(px)(1) - t * t;
                auto d_inner = decltype(px)(0.7978845608028654) * (decltype(px)(1) + decltype(px)(0.134145) * px * px);
                auto pdf = decltype(px)(0.5) * px * cosh_term * d_inner;
                auto cdf = decltype(px)(0.5) * (decltype(px)(1) + t);
                return pdy * (cdf + pdf); 
            });
    }
} // namespace SushiBLAS
