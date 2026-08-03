/**************************************************************************/
/* syr2.cpp                                                               */
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

    sycl::event Level2::syr2(const Tensor& x, const Tensor& y, Tensor& A, bool upper, double alpha)
    {
        SB_THROW_IF(A.rank < 2, "SYR2 requires A to be at least a 2D matrix.");
        SB_THROW_IF(A.dtype != x.dtype || A.dtype != y.dtype, "Data type mismatch in SYR2.");

        int32_t rA = A.rank;
        int64_t n = A.shape[rA - 1]; // Symmetric matrix must be n x n
        SB_THROW_IF(A.shape[rA - 2] != n, "SYR2 requires A to be a square matrix.");

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];

        int64_t nx, incx;
        Internal::get_vec_params(x, nx, incx);

        int64_t ny, incy;
        Internal::get_vec_params(y, ny, incy);

        SB_THROW_IF(nx != n, "Dimension mismatch for vector x in SYR2.");
        SB_THROW_IF(ny != n, "Dimension mismatch for vector y in SYR2.");

        // TODO: Implement multi-dimensional batch support for Level 2

        void* read_x = x.data_or_null();
        void* read_y = y.data_or_null();
        void* rw_A = A.data_or_null();

        // Same as SYR: A is accumulated into and only one triangle is touched, so the region
        // is read-modify-write.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        if (read_y) reads.push_back(y.region());
        if (rw_A) reads.push_back(A.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_A) writes.push_back(A.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl2.syr2";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl2.syr2"_op;
        meta.set_param(0, upper);
        meta.set_param(1, alpha);

        auto layout = A.layout;
        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto px = static_cast<const T*>(read_x);
                auto py = static_cast<const T*>(read_y);
                auto pA = static_cast<T*>(rw_A);
                auto alpha_t = static_cast<T>(alpha);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::syr2<T>(layout, upper, n, alpha_t, px, incx, py, incy, pA, lda); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::syr2<T>(q2, layout, upper, n, alpha_t, px, incx, py, incy, pA, lda, deps2);
                    });
            });
    }
}
