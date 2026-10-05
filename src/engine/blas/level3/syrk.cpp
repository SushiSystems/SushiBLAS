/****************************************************************/
/* syrk.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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

    sycl::event Level3::syrk(const Tensor& A, Tensor& C, 
                      bool upper, bool transA, 
                      double alpha, double beta) 
    {
        // 1. Validation
        SB_THROW_IF(A.rank < 2 || C.rank < 2, "SYRK requires at least 2D tensors.");
        SB_THROW_IF(A.dtype != C.dtype, "Data type mismatch in SYRK operation.");

        int32_t rA = A.rank, rC = C.rank;
        
        // C must be strictly square n x n
        int64_t n = C.shape[rC - 2];
        SB_THROW_IF(C.shape[rC - 1] != n, "SYRK requires C to be a square matrix.");

        // Define inner k based on trans flag
        int64_t inner_n = transA ? A.shape[rA - 1] : A.shape[rA - 2];
        int64_t k       = transA ? A.shape[rA - 2] : A.shape[rA - 1];
        
        SB_THROW_IF(inner_n != n, "Dimension mismatch in SYRK: A does not match C dimension.");

        // Batch extraction
        int64_t batch_size = 1;
        for (int i = 0; i < rC - 2; ++i) batch_size *= C.shape[i];

        // Lda / Ldc
        int64_t lda = (A.layout == Core::Layout::ROW_MAJOR) ? A.shape[rA - 1] : A.shape[rA - 2];
        int64_t ldc = (C.layout == Core::Layout::ROW_MAJOR) ? C.shape[rC - 1] : C.shape[rC - 2];

        // Strides
        int64_t str_a = (batch_size > 1) ? A.strides[rA - 3] : 0;
        int64_t str_c = (batch_size > 1) ? C.strides[rC - 3] : 0;

        // Capture data safely for asynchronous execution
        auto layout = A.layout;
        void* read_A = A.data_or_null();
        void* rw_C = C.data_or_null();

        // C is read-modify-write unconditionally, for two independent reasons: the beta term
        // reads it when beta != 0, and the triangular mask leaves the opposite triangle
        // untouched even when beta == 0. A byte interval cannot express "one triangle", so
        // the honest declaration for the whole region is read *and* write.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_A) reads.push_back(A.region());
        if (rw_C) reads.push_back(C.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_C) writes.push_back(C.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl3.syrk";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl3.syrk"_op;
        meta.set_param(0, alpha);
        meta.set_param(1, beta);
        meta.set_param(2, upper);
        meta.set_param(3, transA);

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, A.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto pA = static_cast<const T*>(read_A);
                auto pC = static_cast<T*>(rw_C);
                auto alpha_t = static_cast<T>(alpha);
                auto beta_t = static_cast<T>(beta);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=]
                    {
                        Kernels::Reference::syrk<T>(layout, upper, transA, n, k, alpha_t, pA, lda, str_a, beta_t, pC, ldc, str_c, batch_size);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::syrk<T>(q2, layout, upper, transA, n, k, alpha_t, pA, lda, str_a, beta_t, pC, ldc, str_c, batch_size, deps2);
                    });
            });
    }
} // namespace SushiBLAS
