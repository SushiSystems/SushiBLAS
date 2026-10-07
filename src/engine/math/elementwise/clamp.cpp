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
#include <SushiBLAS/engine/math/elementwise/scalar_binding.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle ElementwiseOps::clamp(Tensor& t, double min_val, double max_val)
    {
        recorder_.require_sycl_path("Elementwise operation", "math.ew.clamp");
        Dispatch::require_operand(!(min_val > max_val), "math.ew.clamp", "min_val",
                                  "is {}, above max_val {}", min_val, max_val);

        return Dispatch::execute_unary_inplace<Dispatch::Domain::REAL_AND_INTEGRAL>(
            recorder_, t, "math.ew.clamp", "math.ew.clamp"_op,
            Dispatch::bind_scalars([](auto x, auto lo, auto hi) { return sycl::clamp(x, lo, hi); },
                                   Dispatch::ElementScalar(min_val),
                                   Dispatch::ElementScalar(max_val)),
            {min_val, max_val});
    }
} // namespace SushiBLAS
