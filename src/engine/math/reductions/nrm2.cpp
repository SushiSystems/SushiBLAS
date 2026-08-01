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
#include "red_internal.hpp"

namespace SushiBLAS 
{
    sycl::event ReductionOps::nrm2(const Tensor& t, Tensor& result) 
    {
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps) 
        {
            auto e1 = q.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size), sycl::reduction(pR, T(0), sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}), [=](sycl::id<1> idx, auto& reducer) 
                {
                    T val = pT[idx[0]];
                    reducer.combine(val * val);
                });
            });
            return q.submit([&](sycl::handler& h) 
            {
                h.depends_on(e1);
                h.single_task([=]() 
                {
                    *pR = sycl::sqrt(*pR);
                });
            });
        };
        return Internal::execute_reduction(engine_, t, result, "math.nrm2", "math.nrm2"_op, dispatch);
    }
} // namespace SushiBLAS
