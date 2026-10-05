/****************************************************************/
/* div.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file div.cpp
 * @brief Defines ElementwiseOps::div, which records the elementwise div task over two tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::div(const Tensor& A, const Tensor& B, Tensor& C) 
    {
        return Dispatch::execute_binary<Dispatch::Domain::REAL_AND_COMPLEX>(engine_, A, B, C, "math.ew.div", "math.ew.div"_op, 
            [](auto x, auto y) { return x / y; });
    }
} // namespace SushiBLAS
