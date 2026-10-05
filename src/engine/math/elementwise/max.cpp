/****************************************************************/
/* max.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::max(const Tensor& A, const Tensor& B, Tensor& C) 
    {
        return Dispatch::execute_binary(engine_, A, B, C, "math.ew.max", "math.ew.max"_op, 
            [](auto x, auto y) { return sycl::max(x, y); });
    }
} // namespace SushiBLAS
