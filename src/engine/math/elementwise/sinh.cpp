/****************************************************************/
/* sinh.cpp                                                     */
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

    sycl::event ElementwiseOps::sinh(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.sinh", "math.ew.sinh"_op, 
            [](auto x) { return sycl::sinh(x); });
    }
} // namespace SushiBLAS
