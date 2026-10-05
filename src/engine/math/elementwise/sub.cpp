/****************************************************************/
/* sub.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file sub.cpp
 * @brief Defines ElementwiseOps::sub, which records the elementwise sub task over two tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::sub(const Tensor& A, const Tensor& B, Tensor& C) 
    {
        return Dispatch::execute_binary<Dispatch::Domain::REAL_AND_COMPLEX>(engine_, A, B, C, "math.ew.sub", "math.ew.sub"_op, 
            [](auto x, auto y) { return x - y; });
    }
} // namespace SushiBLAS
