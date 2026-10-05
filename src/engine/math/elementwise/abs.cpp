/****************************************************************/
/* abs.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file abs.cpp
 * @brief Defines ElementwiseOps::abs, which records the in-place elementwise abs task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::abs(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace<Dispatch::Domain::REAL_AND_COMPLEX>(engine_, t, "math.ew.abs", "math.ew.abs"_op, 
            [](auto x) { return Dispatch::safe_abs(x); });
    }
} // namespace SushiBLAS
