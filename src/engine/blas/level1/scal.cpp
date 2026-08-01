/**************************************************************************/
/* scal.cpp                                                               */
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

#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include "level1_internal.hpp"

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level1::scal(float alpha, Tensor& x) 
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);

        // TODO: Implement multi-dimensional batch support

        void* write_x = x.storage ? x.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        std::vector<void*> writes = {};
        if (write_x) writes.push_back(write_x);

        return Internal::execute_level1(engine_, "blas.lvl1.scal", "blas.lvl1.scal"_op, x.dtype, reads, writes, {alpha},
            [n, alpha, incx, pX=write_x](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (std::is_same_v<T, sycl::half>) 
                {
                    SB_THROW_IF(true, "MKL Level 1 BLAS does not support HALF precision natively.");
                    return sycl::event();
                } 
                else 
                {
                    T alpha_t;
                    if constexpr (Internal::is_complex_v<T>)
                        alpha_t = T(alpha, 0.0);
                    else
                        alpha_t = static_cast<T>(alpha);
                    
                    SB_LOG_INFO("MKL SCAL: {} elements", n);
                    return oneapi::mkl::blas::column_major::scal(q, n, alpha_t, static_cast<T*>(pX), incx, deps);
                }
            });
    }
}
