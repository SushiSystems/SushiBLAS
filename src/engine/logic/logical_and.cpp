/****************************************************************/
/* logical_and.cpp                                              */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file logical_and.cpp
 * @brief Defines LogicOps::logical_and, which records the elementwise logical_and task over two
 * tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiBLAS/engine/logic/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event LogicOps::logical_and(const Tensor& A, const Tensor& B, Tensor& result) 
    {
        return Dispatch::execute_logic_binary<Dispatch::Domain::ALL>(engine_, A, B, result, "logic.and", "logic.and"_op, 
            [](auto x, auto y) { return (Dispatch::logic_not_zero(x) && Dispatch::logic_not_zero(y)) ? decltype(x)(1) : decltype(x)(0); });
    }
} // namespace SushiBLAS
