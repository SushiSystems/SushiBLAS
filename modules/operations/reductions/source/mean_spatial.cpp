/**************************************************************************/
/* mean_spatial.cpp                                                       */
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
 * @file mean_spatial.cpp
 * @brief Defines ReductionOps::mean_spatial, which records the spatial mean reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    // The fold is the one mean uses, over a span keyed by (n, c); see modules/operations/README.md.
    TaskHandle ReductionOps::mean_spatial(const Tensor& t, Tensor& result)
    {
        Dispatch::require_min_elements(t, 1, "math.mean_spatial", "t");

        const Reduce::AxisSpan span =
            Dispatch::spatial_span_for(t, result, "math.mean_spatial");

        return Dispatch::execute_folding_span_reduction<Reduce::CompensatedSum>(
            recorder_, t, span, result, "math.mean_spatial", "math.mean_spatial"_op,
            Dispatch::ReduceIdentityTransform{},
            [](auto value, std::size_t n)
            {
                using T = decltype(value);
                return value / static_cast<T>(n);
            });
    }
} // namespace SushiBLAS
