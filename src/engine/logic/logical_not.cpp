/**************************************************************************/
/* logical_not.cpp                                                        */
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
 * @file logical_not.cpp
 * @brief Defines LogicOps::logical_not, which records the elementwise logical_not task.
 * @author Mustafa Garip
 */

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
