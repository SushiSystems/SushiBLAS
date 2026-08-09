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

    // The same three pieces `mean` is - compensated sum, nothing to transform,
    // `v / n` as the finalizer - over the segmented seam rather than the
    // whole-tensor one. All that is new is the span: a fold identified by a pair
    // `(n, c)`, which is what `spatial_span_for` builds and what AxisSpan's second
    // segment axis exists to carry.
    sycl::event ReductionOps::mean_spatial(const Tensor& t, Tensor& result)
    {
        const Reduce::AxisSpan span =
            Dispatch::spatial_span_for(t, result, "math.mean_spatial");

        return Dispatch::execute_folding_span_reduction<Reduce::CompensatedSum>(
            engine_, t, span, result, "math.mean_spatial", "math.mean_spatial"_op,
            Dispatch::ReduceIdentityTransform{},
            [](auto value, std::size_t n)
            {
                using T = decltype(value);
                return value / static_cast<T>(n);
            });
    }
} // namespace SushiBLAS
