/**************************************************************************/
/* max.cpp                                                                */
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

#include <limits>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    // On sycl::reduction deliberately - this is not an oversight sitting next to the
    // summation reductions. The SYCL spec does not fix the order in which a reduction
    // combines its partials, which is a real hazard for `+`, since floating-point addition
    // is not associative and the combination order is therefore visible in the result's
    // bits and free to change with the launch configuration. `maximum` has no such
    // exposure: it is exactly associative and exactly commutative, so every combination
    // order yields the same value and a hand-written fixed-order tree would buy nothing
    // but slower code.
    //
    // One caveat, and it is a property of the input rather than of the launch geometry:
    // sycl::maximum selects with `<`, and every comparison against a NaN is false, so which
    // operand survives an encounter with NaN does depend on the order. No reduction shape
    // available here fixes that; only screening the input does.
    sycl::event ReductionOps::max(const Tensor& t, Tensor& result)
    {
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size,
                                     [[maybe_unused]] Dispatch::ReductionScratch<T> scratch,
                                     const std::vector<sycl::event>& deps) 
        {
            T init_val = std::numeric_limits<T>::lowest();
            return q.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size), sycl::reduction(pR, init_val, sycl::maximum<T>(), sycl::property::reduction::initialize_to_identity{}), [=](sycl::id<1> idx, auto& reducer) 
                {
                    reducer.combine(pT[idx[0]]);
                });
            });
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.max", "math.max"_op, dispatch);
    }
} // namespace SushiBLAS
