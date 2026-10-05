/****************************************************************/
/* where.cpp                                                    */
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

    sycl::event LogicOps::where(const Tensor& condition, const Tensor& A, const Tensor& B, Tensor& out) 
    {
        return Dispatch::execute_logic_ternary<Dispatch::Domain::ALL>(engine_, condition, A, B, out, "logic.where", "logic.where"_op, 
            [](auto c, auto a, auto b) { return Dispatch::logic_not_zero(c) ? a : b; });
    }
} // namespace SushiBLAS
