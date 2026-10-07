/**************************************************************************/
/* min.cpp                                                                */
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
 * @file min.cpp
 * @brief Defines ElementwiseOps::min, which records the elementwise min task over two tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/elementwise.hpp>
#include <type_traits>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle ElementwiseOps::min(const Tensor& A, const Tensor& B, Tensor& C)
    {
        return Dispatch::execute_binary<Dispatch::Domain::REAL_AND_INTEGRAL>(
            recorder_, A, B, C, "math.ew.min", "math.ew.min"_op,
            [](auto x, auto y)
            {
                using T = decltype(x);
                if constexpr (std::is_integral_v<T>)
                    return sycl::min(x, y);
                else
                    return (x != x) ? x : ((y != y) ? y : sycl::min(x, y));
            });
    }
} // namespace SushiBLAS
