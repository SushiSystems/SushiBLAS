/**************************************************************************/
/* nrm2.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // Square the elements going in, take the root coming out. Like `mean`, the
    // finalizer folds into the last kernel instead of costing a second launch.
    //
    // Still the naive 2-norm rather than the scaling one BLAS's xNRM2 specifies:
    // squaring overflows for inputs above ~1e19 in fp64 and ~1e19 in fp32 where the
    // scaled form would not. That is pre-existing and out of scope here; the
    // compensated fold makes the *summation* accurate, not the squaring safe.
    sycl::event ReductionOps::nrm2(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
            engine_, t, result, "math.nrm2", "math.nrm2"_op,
            [](auto x) { return x * x; },
            [](auto value, [[maybe_unused]] std::size_t n) { return sycl::sqrt(value); });
    }
} // namespace SushiBLAS
