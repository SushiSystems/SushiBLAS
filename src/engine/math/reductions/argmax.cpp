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

#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include <limits>
#include <type_traits>
#include <SushiBLAS/engine/math/reductions.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        struct ArgMaxPair
        {
            T val;
            int64_t idx;
        };

        template<typename T>
        struct ArgMaxCombine
        {
            ArgMaxPair<T> operator()(const ArgMaxPair<T>& a, const ArgMaxPair<T>& b) const
            {
                if (a.val > b.val) return a;
                if (a.val < b.val) return b;
                return (a.idx < b.idx) ? a : b;
            }
        };
    }

    sycl::event ReductionOps::argmax(const Tensor& t, Tensor& result)
    {
        // Already order-independent by construction: a sequential scan visits indices in
        // ascending order every time, so the lowest-index maximum always wins the tie and
        // the answer is reproducible to the bit. Whatever replaces this must keep that -
        // ArgMaxCombine above breaks ties on the index precisely so a tree reduction would
        // stay equivalent.
        //
        // TODO: Implement a parallel tree reduction for argmax to replace single_task.
        // A single-threaded device loop for now: custom reductions through sycl::reduction
        // behave differently across the OpenCL and Level-Zero backends, and this op's
        // tie-breaking rule is the part that must not move.
        //
        // The result goes into an INT64 tensor through execute_index_reduction, rather than
        // into a same-dtype tensor through execute_reduction. It used to be stored as
        // `static_cast<T>(idx)` - a float. Exact below 2^24, and above that two adjacent
        // indices round to the same value, so the op returned a confident wrong answer on
        // exactly the tensor sizes a training run calls it on. In HALF the ceiling is 2048.
        auto dispatch = [](sycl::queue& q, const auto* pT, int64_t* pR, int64_t size,
                           const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;

            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.single_task([=]()
                {
                    if (size == 0) return;
                    T max_val = pT[0];
                    int64_t max_idx = 0;
                    for (int64_t i = 1; i < size; ++i)
                    {
                        if (pT[i] > max_val)
                        {
                            max_val = pT[i];
                            max_idx = i;
                        }
                    }
                    *pR = max_idx;
                });
            });
        };
        return Dispatch::execute_index_reduction(engine_, t, result, "math.argmax", "math.argmax"_op, dispatch);
    }
} // namespace SushiBLAS
