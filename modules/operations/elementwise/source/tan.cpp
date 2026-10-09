/**************************************************************************/
/* tan.cpp                                                                */
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
 * @file tan.cpp
 * @brief Defines ElementwiseOps::tan, which records the in-place elementwise tan task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle ElementwiseOps::tan(Tensor& t) 
    {
        return Dispatch::execute_unary_inplace(recorder_, t, "math.ew.tan", "math.ew.tan"_op, 
            [](auto x) { return sycl::tan(x); });
    }
} // namespace SushiBLAS
