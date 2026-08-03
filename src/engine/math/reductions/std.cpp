/**************************************************************************/
/* std.cpp                                                                */
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

#include <type_traits>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include <SushiBLAS/engine/math/reductions.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ReductionOps::std(const Tensor& t, Tensor& result) 
    {
        // TODO: Implement a parallel reduction (e.g. Chan et al. algorithm) for standard deviation to replace single_task.
        auto dispatch = [](sycl::queue& q, const auto* pT, auto* pR, int64_t size,
                           [[maybe_unused]] auto scratch,
                           const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;

            return q.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.single_task([=]() 
                {
                    if (size == 0) return;
                    T mean = 0;
                    T M2 = 0;
                    for (int64_t i = 0; i < size; ++i) 
                    {
                        T val = pT[i];
                        T delta = val - mean;
                        mean += delta / static_cast<T>(i + 1);
                        T delta2 = val - mean;
                        M2 += delta * delta2;
                    }
                    *pR = sycl::sqrt(M2 / static_cast<T>(size)); // Population std
                });
            });
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.std", "math.std"_op, dispatch);
    }
} // namespace SushiBLAS
