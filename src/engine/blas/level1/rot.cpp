/**************************************************************************/
/* rot.cpp                                                                */
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

    sycl::event Level1::rot(Tensor& x, Tensor& y, float c, float s) 
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in ROT.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "ROT requires tensors of the same number of elements.");

        // TODO: Implement multi-dimensional batch support

        void* write_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_y = y.storage ? y.storage->data_ptr : nullptr;

        std::vector<void*> reads = {}; // x and y are updated
        std::vector<void*> writes = {};
        if (write_x) writes.push_back(write_x);
        if (write_y) writes.push_back(write_y);

        return Internal::execute_level1(engine_, "blas.lvl1.rot", "blas.lvl1.rot"_op, x.dtype, reads, writes, {c, s},
            [n, c, s, incx, incy, pX=write_x, pY=write_y](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (std::is_same_v<T, sycl::half>) 
                {
                    SB_THROW_IF(true, "MKL Level 1 BLAS does not support HALF precision natively.");
                    return sycl::event();
                } 
                else 
                {
                    if constexpr (Internal::is_complex_v<T>)
                        return oneapi::mkl::blas::column_major::rot(q, n, static_cast<T*>(pX), incx, static_cast<T*>(pY), incy, static_cast<typename T::value_type>(c), static_cast<typename T::value_type>(s), deps);
                    else 
                        return oneapi::mkl::blas::column_major::rot(q, n, static_cast<T*>(pX), incx, static_cast<T*>(pY), incy, static_cast<T>(c), static_cast<T>(s), deps);
                }
            });
    }
}
