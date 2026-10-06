/**************************************************************************/
/* clamp.cpp                                                              */
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
 * @file clamp.cpp
 * @brief Defines ElementwiseOps::clamp, which records the in-place elementwise clamp task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ElementwiseOps::clamp(Tensor& t, float min_val, float max_val) 
    {
        return Dispatch::execute_unary_inplace(engine_, t, "math.ew.clamp", "math.ew.clamp"_op, 
            [min_val, max_val](auto x) { return sycl::clamp(x, (decltype(x))min_val, (decltype(x))max_val); },
            {min_val, max_val});
    }
} // namespace SushiBLAS
