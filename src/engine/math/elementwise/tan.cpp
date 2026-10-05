/****************************************************************/
/* tan.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file tan.cpp
 * @brief Defines ElementwiseOps::tan, which records the in-place elementwise tan task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::tan(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.tan", "math.ew.tan"_op, 
            [](auto x) { return sycl::tan(x); });
    }
} // namespace SushiBLAS
