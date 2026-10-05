/****************************************************************/
/* square.cpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file square.cpp
 * @brief Defines ElementwiseOps::square, which records the in-place elementwise square task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::square(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace<Dispatch::Domain::REAL_AND_COMPLEX>(engine_, t, "math.ew.square", "math.ew.square"_op, 
            [](auto x) { return x * x; });
    }
} // namespace SushiBLAS
