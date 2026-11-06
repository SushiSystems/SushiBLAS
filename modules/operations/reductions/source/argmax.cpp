/**************************************************************************/
/* argmax.cpp                                                             */
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
 * @file argmax.cpp
 * @brief Defines ReductionOps::argmax, which records the argmax reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiBLAS/reductions/arg_reduction.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle ReductionOps::argmax(const Tensor& t, Tensor& result)
    {
        Dispatch::require_min_elements(t, 1, "math.argmax", "t");
        Dispatch::require_storage(t, "math.argmax", "t");

        return Dispatch::execute_index_reduction(
            recorder_, t, result, "math.argmax", "math.argmax"_op,
            Dispatch::ArgReductionDispatch<Reduce::RunningArgMax>{},
            Dispatch::planned_partials_for(t));
    }
} // namespace SushiBLAS
