/**************************************************************************/
/* logical_and.cpp                                                        */
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
 * @file logical_and.cpp
 * @brief Defines LogicOps::logical_and, which records the elementwise logical_and task over two
 * tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/logic/logic.hpp>
#include <SushiBLAS/logic/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Literals;

    TaskHandle LogicOps::logical_and(const Tensor& A, const Tensor& B, Tensor& result) 
    {
        return Dispatch::execute_logic_binary<Dispatch::Domain::All>(recorder_, A, B, result, "logic.and", "logic.and"_op, 
            [](auto x, auto y) { return (Dispatch::logic_not_zero(x) && Dispatch::logic_not_zero(y)) ? decltype(x)(1) : decltype(x)(0); });
    }
} // namespace SushiBLAS
