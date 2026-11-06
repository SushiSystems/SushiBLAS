/**************************************************************************/
/* asum.cpp                                                               */
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
 * @file asum.cpp
 * @brief Defines ReductionOps::asum, which records the sum-of-magnitudes reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    // Sum of magnitudes: the same fold as `sum` with `fabs` as the transform.
    TaskHandle ReductionOps::asum(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
            recorder_, t, result, "math.asum", "math.asum"_op,
            [](auto x) { return sycl::fabs(x); },
            Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
