/**************************************************************************/
/* gemv.cpp                                                               */
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
#include <SushiBLAS/engine/blas/level2.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level2::gemv(const Tensor& A, const Tensor& x, Tensor& y,
                             bool transA, float alpha, float beta)
    {
        SB_THROW_IF(A.rank < 2, "GEMV requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in GEMV.");

        int32_t rA = A.rank;
        int64_t m = A.shape[rA - 2];
        int64_t n = A.shape[rA - 1];

        // TODO: Implement multi-dimensional batch support for Level 2 GEMV

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);

        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        int64_t expected_nx = transA ? m : n;
        int64_t expected_ny = transA ? n : m;

        SB_THROW_IF(nx != expected_nx, "Dimension mismatch for vector x in GEMV.");
        SB_THROW_IF(ny != expected_ny, "Dimension mismatch for vector y in GEMV.");

        void* read_A = A.storage ? A.storage->data_ptr : nullptr;
        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_y = y.storage ? y.storage->data_ptr : nullptr;

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(read_A);
        if (read_x) reads.push_back(read_x);
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_y) writes.push_back(write_y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.gemv";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.gemv"_op;
        meta.set_param(0, transA);
        meta.set_param(1, alpha);
        meta.set_param(2, beta);

        auto layout = A.layout;
        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pA = static_cast<const T*>(read_A);
                auto px = static_cast<const T*>(read_x);
                auto py = static_cast<T*>(write_y);
                auto alpha_t = static_cast<T>(alpha);
                auto beta_t = static_cast<T>(beta);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::gemv<T>(layout, transA, m, n, alpha_t, pA, lda, px, incx, beta_t, py, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::gemv<T>(q2, layout, transA, m, n, alpha_t, pA, lda, px, incx, beta_t, py, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
