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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file min.cpp
 * @brief Defines ElementwiseOps::min, which records the elementwise min task over two tensors.
 * @author Mustafa Garip
 */

#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/elementwise/scalar_math.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle ElementwiseOps::min(const Tensor& A, const Tensor& B, Tensor& C)
    {
        return Dispatch::execute_binary<Dispatch::Domain::RealAndIntegral>(
            recorder_, A, B, C, "math.ew.min", "math.ew.min"_op,
            [](auto x, auto y) { return Dispatch::scalar_min(x, y); });
    }
} // namespace SushiBLAS
