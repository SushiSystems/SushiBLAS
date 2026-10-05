/****************************************************************/
/* acosh.cpp                                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file acosh.cpp
 * @brief Defines ElementwiseOps::acosh, which records the in-place elementwise acosh task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::acosh(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.acosh", "math.ew.acosh"_op, 
            [](auto x) { return sycl::acosh(x); });
    }
} // namespace SushiBLAS
