/****************************************************************/
/* logical_not.cpp                                              */
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

    sycl::event LogicOps::logical_not(const Tensor& A, Tensor& result) 
    {
        return Dispatch::execute_logic_unary<Dispatch::Domain::ALL>(engine_, A, result, "logic.not", "logic.not"_op, 
            [](auto x) { return Dispatch::logic_not_zero(x) ? decltype(x)(0) : decltype(x)(1); });
    }
} // namespace SushiBLAS
