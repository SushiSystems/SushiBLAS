/****************************************************************/
/* ceil.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file ceil.cpp
 * @brief Defines ElementwiseOps::ceil, which records the in-place elementwise ceil task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::ceil(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.ceil", "math.ew.ceil"_op, 
            [](auto x) { return sycl::ceil(x); });
    }
} // namespace SushiBLAS
