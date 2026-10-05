/****************************************************************/
/* less.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiBLAS/engine/logic/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event LogicOps::less(const Tensor& A, const Tensor& B, Tensor& result) 
    {
        return Dispatch::execute_logic_binary<Dispatch::Domain::REAL_AND_INTEGRAL>(engine_, A, B, result, "logic.less", "logic.less"_op, 
            [](auto x, auto y) { return Dispatch::logic_less(x, y) ? decltype(x)(1) : decltype(x)(0); });
    }
} // namespace SushiBLAS
