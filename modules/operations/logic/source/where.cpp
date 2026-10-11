/**************************************************************************/
/* where.cpp                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file where.cpp
 * @brief Defines LogicOps::where, which records the elementwise choice between two tensors by a
 * condition.
 * @author Mustafa Garip
 */

#include <SushiBLAS/logic/logic.hpp>
#include <SushiBLAS/logic/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Literals;

    TaskHandle LogicOps::where(const Tensor& condition, const Tensor& A, const Tensor& B, Tensor& out) 
    {
        return Dispatch::execute_logic_ternary<Dispatch::Domain::All>(recorder_, condition, A, B, out, "logic.where", "logic.where"_op, 
            [](auto c, auto a, auto b) { return Dispatch::logic_not_zero(c) ? a : b; });
    }
} // namespace SushiBLAS
