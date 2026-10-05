/****************************************************************/
/* remainder.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file remainder.cpp
 * @brief Defines ElementwiseOps::remainder, which records the elementwise remainder task over two
 * tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::remainder(const Tensor& A, const Tensor& B, Tensor& C) 
    {
        return Dispatch::execute_binary(engine_, A, B, C, "math.ew.remainder", "math.ew.remainder"_op, 
            [](auto x, auto y) { return sycl::remainder(x, y); });
    }
} // namespace SushiBLAS
