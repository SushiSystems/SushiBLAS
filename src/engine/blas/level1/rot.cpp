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
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiRuntime/graph/task_types.hpp>

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

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_x) writes.push_back(write_x);
        if (write_y) writes.push_back(write_y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.rot";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.rot"_op;
        meta.set_param(0, c);
        meta.set_param(1, s);

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                using Real = typename Kernels::Reference::RealOf<T>::type;
                auto pX = static_cast<T*>(write_x);
                auto pY = static_cast<T*>(write_y);
                auto c_t = static_cast<Real>(c);
                auto s_t = static_cast<Real>(s);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::rot<T>(n, pX, incx, pY, incy, c_t, s_t); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::rot<T>(q2, n, pX, incx, pY, incy, c_t, s_t, deps2);
                    });
            });
    }
}
