/**************************************************************************/
/* mul.cpp                                                                */
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
 * @file mul.cpp
 * @brief Defines ElementwiseOps::mul, which records the elementwise mul task over two tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/elementwise/wrapping_math.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Literals;

    TaskHandle ElementwiseOps::mul(const Tensor& A, const Tensor& B, Tensor& C) 
    {
        return Dispatch::execute_binary<Dispatch::Domain::All>(recorder_, A, B, C, "math.ew.mul", "math.ew.mul"_op, 
            [](auto x, auto y) { return Dispatch::wrapping_mul(x, y); });
    }
} // namespace SushiBLAS
