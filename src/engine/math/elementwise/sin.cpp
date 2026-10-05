/****************************************************************/
/* sin.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file sin.cpp
 * @brief Defines ElementwiseOps::sin, which records the in-place elementwise sin task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::sin(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.sin", "math.ew.sin"_op, 
            [](auto x) { return sycl::sin(x); });
    }
} // namespace SushiBLAS
