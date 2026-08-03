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
        // TODO: Implement a parallel tree reduction for argmax to replace single_task.
        // For argmax, result tensor actually stores the index. Since the reduction dispatch assumes 
        // type T for pR, we will use a temporary array or variable, but we can also just 
        // fall back to a simpler approach or allocate a small struct.
        // For now, to keep the reduction dispatch clean and avoid type mismatches, we'll implement 
        // a simple single-threaded loop on the device, as Custom Reductions with sycl::reduction 
        // can be tricky in OpenCL vs Level-Zero backends.
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps) 
        {
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
                    *pR = static_cast<T>(max_idx);
                });
            });
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.argmax", "math.argmax"_op, dispatch);
    }
} // namespace SushiBLAS
