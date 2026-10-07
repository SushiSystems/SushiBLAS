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
#include <SushiBLAS/engine/math/elementwise/scalar.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle ElementwiseOps::clamp(Tensor& t, double min_val, double max_val)
    {
        SB_THROW_IF(min_val > max_val,
                    "Elementwise clamp: min_val {} is greater than max_val {}.", min_val, max_val);
        const Dispatch::ElementScalar lo(min_val);
        const Dispatch::ElementScalar hi(max_val);
        return Dispatch::execute_unary_inplace<Dispatch::Domain::REAL_AND_INTEGRAL>(
            recorder_, t, "math.ew.clamp", "math.ew.clamp"_op,
            [lo, hi](auto x)
            {
                using T = decltype(x);
                return sycl::clamp(x, lo.as<T>(), hi.as<T>());
            },
            {min_val, max_val});
    }
} // namespace SushiBLAS
