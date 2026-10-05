/****************************************************************/
/* atanh.cpp                                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file atanh.cpp
 * @brief Defines ElementwiseOps::atanh, which records the in-place elementwise atanh task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::atanh(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.atanh", "math.ew.atanh"_op, 
            [](auto x) { return sycl::atanh(x); });
    }
} // namespace SushiBLAS
