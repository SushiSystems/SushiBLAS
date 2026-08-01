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

#include "red_internal.hpp"
#include <SushiBLAS/engine/math/reductions.hpp>

namespace SushiBLAS 
{
    sycl::event ReductionOps::std(const Tensor& t, Tensor& result) 
    {
        // TODO: Implement a parallel reduction (e.g. Chan et al. algorithm) for standard deviation to replace single_task.
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps) 
        {
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
        return Internal::execute_reduction(engine_, t, result, "math.std", "math.std"_op, dispatch);
    }
} // namespace SushiBLAS
