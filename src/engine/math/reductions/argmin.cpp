/**************************************************************************/
/* argmin.cpp                                                             */
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

    sycl::event ReductionOps::argmin(const Tensor& t, Tensor& result)
    {
        // Already order-independent by construction: a sequential scan visits indices in
        // ascending order every time, so the lowest-index minimum always wins the tie and
        // the answer is reproducible to the bit. Whatever replaces this must keep that -
        // a tree reduction is only equivalent if its combiner breaks ties on the index,
        // the way ArgMaxCombine in argmax.cpp does.
        //
        // TODO: Implement a parallel tree reduction for argmin to replace single_task.
        //
        // The result goes into an INT64 tensor through execute_index_reduction, rather than
        // into a same-dtype tensor through execute_reduction. It used to be stored as
        // `static_cast<T>(idx)` - a float. Exact below 2^24, and above that two adjacent
        // indices round to the same value, so the op returned a confident wrong answer on
        // exactly the tensor sizes a training run calls it on. In HALF the ceiling is 2048.
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, int64_t* pR, int64_t size,
                                       const std::vector<sycl::event>& deps)
        {
            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.single_task([=]()
                {
                    if (size == 0) return;
                    T min_val = pT[0];
                    int64_t min_idx = 0;
                    for (int64_t i = 1; i < size; ++i)
                    {
                        if (pT[i] < min_val)
                        {
                            min_val = pT[i];
                            min_idx = i;
                        }
                    }
                    *pR = min_idx;
                });
            });
        };
        return Dispatch::execute_index_reduction(engine_, t, result, "math.argmin", "math.argmin"_op, dispatch);
    }
} // namespace SushiBLAS
