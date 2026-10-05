/****************************************************************/
/* less_equal.cpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file less_equal.cpp
 * @brief Defines LogicOps::less_equal, which records the elementwise less_equal task over two
 * tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiBLAS/engine/logic/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event LogicOps::less_equal(const Tensor& A, const Tensor& B, Tensor& result) 
    {
        return Dispatch::execute_logic_binary<Dispatch::Domain::REAL_AND_INTEGRAL>(engine_, A, B, result, "logic.less_equal", "logic.less_equal"_op, 
            [](auto x, auto y) { return Dispatch::logic_less_equal(x, y) ? decltype(x)(1) : decltype(x)(0); });
    }
} // namespace SushiBLAS
