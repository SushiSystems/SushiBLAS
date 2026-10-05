/****************************************************************/
/* add.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file add.cpp
 * @brief Defines ElementwiseOps::add, which records the elementwise add task over two tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::add(const Tensor& A, const Tensor& B, Tensor& C) 
    {
        return Dispatch::execute_binary<Dispatch::Domain::REAL_AND_COMPLEX>(engine_, A, B, C, "math.ew.add", "math.ew.add"_op, 
            [](auto x, auto y) { return x + y; });
    }
} // namespace SushiBLAS
