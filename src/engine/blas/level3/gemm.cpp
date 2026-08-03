/**************************************************************************/
/* gemm.cpp                                                               */
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

#include <vector>
#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level3::gemm(const Tensor& A, const Tensor& B, Tensor& C, 
                            bool transA, bool transB,
                            double alpha, double beta) 
    {
        // 1. Validation
        SB_THROW_IF(A.rank < 2 || B.rank < 2 || C.rank < 2, "GEMM requires at least 2D tensors.");
        SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype, "Data type mismatch in GEMM operation.");

        int32_t rA = A.rank, rB = B.rank, rC = C.rank;
        int64_t m = C.shape[rC - 2];
        int64_t n = C.shape[rC - 1];
        int64_t k = transA ? A.shape[rA - 2] : A.shape[rA - 1];

        // Batch & Stride calculation
        int64_t batch_size = 1;
        for (int i = 0; i < rC - 2; ++i) batch_size *= C.shape[i];

        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
        int64_t ldb = (B.layout == Core::Layout::ROW_MAJOR) ? B.shape[rB - 1] : B.shape[rB - 2];
        int64_t ldc = (C.layout == Core::Layout::ROW_MAJOR) ? C.shape[rC - 1] : C.shape[rC - 2];

        int64_t str_a = (batch_size > 1) ? A.strides[rA - 3] : 0;
        int64_t str_b = (batch_size > 1) ? B.strides[rB - 3] : 0;
        int64_t str_c = (batch_size > 1) ? C.strides[rC - 3] : 0;

        // Capture data safely for asynchronous execution
        auto layout = A.layout;
        void* read_A = A.data_or_null();
        void* read_B = B.data_or_null();
        void* rw_C = C.data_or_null();

        // `C = alpha * op(A) * op(B) + beta * C` reads C whenever beta is non-zero - the
        // kernels branch around the read on exactly `beta == 0`. Declaring that read makes
        // the access set true rather than merely survivable via the tracker's WAW edge.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(A.region());
        if (read_B) reads.push_back(B.region());
        if (rw_C && beta != 0.0) reads.push_back(C.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_C) writes.push_back(C.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl3.gemm";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl3.gemm"_op;
        meta.set_param(0, alpha);
        meta.set_param(1, beta);
        meta.set_param(2, transA);
        meta.set_param(3, transB);

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto pA = static_cast<const T*>(read_A);
                auto pB = static_cast<const T*>(read_B);
                auto pC = static_cast<T*>(rw_C);
                auto alpha_t = static_cast<T>(alpha);
                auto beta_t = static_cast<T>(beta);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=]
                    {
                        Kernels::Reference::gemm<T>(layout, transA, transB, m, n, k, alpha_t, pA, lda, str_a, pB, ldb, str_b, beta_t, pC, ldc, str_c, batch_size);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::gemm<T>(q2, layout, transA, transB, m, n, k, alpha_t, pA, lda, str_a, pB, ldb, str_b, beta_t, pC, ldc, str_c, batch_size, deps2);
                    });
            });
    }
} // namespace SushiBLAS
