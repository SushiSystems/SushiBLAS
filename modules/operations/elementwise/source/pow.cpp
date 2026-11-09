/**************************************************************************/
/* pow.cpp                                                                */
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
 * @file pow.cpp
 * @brief Defines ElementwiseOps::pow, which records the in-place elementwise pow task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/elementwise/scalar_math.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Literals;

    TaskHandle ElementwiseOps::pow(Tensor& t, double exponent) 
    {
        return Dispatch::execute_unary_inplace(recorder_, t, "math.ew.pow", "math.ew.pow"_op,
            Dispatch::bind_scalars([](auto x, auto e) { return Dispatch::scalar_pow(x, e); },
                                   Dispatch::ElementScalar(exponent)),
            {exponent});
    }
} // namespace SushiBLAS
