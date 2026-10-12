/**************************************************************************/
/* spmv.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file spmv.cpp
 * @brief Defines SparseOps::spmv, which records the sparse matrix-vector product task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/sparse/sparse.hpp>

#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/sparse_kernels/sparse.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/sparse/accumulator_size.hpp>
#include <SushiBLAS/sparse/sparse_matrix_operand.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Names the operation in its messages and in its task. */
        constexpr const char* K_OP = "sparse.spmv";

        /**
         * @brief Records the SPMV task of a CSR matrix whose indices have type Index.
         * @tparam Index int32_t or int64_t.
         * @pre The operands have passed the checks of SparseOps::spmv.
         */
        template<typename Index>
        TaskHandle record_spmv(TaskRecorder& recorder, const SparseMatrix& A,
                               const Dispatch::SparseMatrixOperand& a, const Tensor& x,
                               int64_t incx, Tensor& y, int64_t incy, Core::Transpose trans,
                               BLASScalar alpha, BLASScalar beta)
        {
            OperandSet operands;
            operands.reads(A.row_offsets).reads(A.column_indices).reads(A.values).reads(x);
            if (beta.is_zero())
                operands.writes(y);
            else
                operands.updates(y);

            // Only a transposed product keeps partial sums: blocks of them, each as long as y.
            const std::size_t blocks = Core::is_transposed(trans)
                ? Kernels::SYCL::scatter_blocks(Device::profile_for(A.row_offsets.device()),
                                                a.cols, a.nnz)
                : std::size_t{0};
            const std::size_t partial_bytes = blocks * static_cast<std::size_t>(a.cols) *
                                              Internal::accumulator_bytes(A.values.dtype);
            const ScratchBuffer scratch =
                Internal::allocate_blas_scratch(recorder, A.row_offsets, partial_bytes, operands);

            const Internal::BLASMetadata metadata = Internal::BLASMetadata(K_OP, "sparse.spmv"_op)
                                                    .scalar(alpha).scalar(beta).flag(trans);

            const int64_t rows = a.rows;
            const int64_t cols = a.cols;
            const int64_t nnz = a.nnz;
            const int64_t first_index = a.first_index;
            const void* read_offsets = A.row_offsets.data_or_null();
            const void* read_columns = A.column_indices.data_or_null();
            const void* read_values = A.values.data_or_null();
            const void* read_x = x.data_or_null();
            void* rw_y = y.data_or_null();

            return Internal::execute_typed_task(recorder, metadata.get(), operands, A.values.dtype,
                                                "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    const Kernels::CsrPattern<Index> pattern{
                        rows, cols, nnz, first_index, static_cast<const Index*>(read_offsets),
                        static_cast<const Index*>(read_columns)};
                    auto pv = static_cast<const T*>(read_values);
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<T*>(rw_y);
                    const T alpha_t = alpha.as<T>();
                    const T beta_t = beta.as<T>();
                    auto partials = static_cast<Kernels::Accumulator<T>*>(scratch.data());

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::spmv<T, Index>(trans, pattern, pv, alpha_t, px,
                                                               incx, beta_t, py, incy);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::spmv<T, Index>(q2, trans, pattern, pv, alpha_t,
                                                                 px, incx, beta_t, py, incy,
                                                                 partials, blocks, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle SparseOps::spmv(const SparseMatrix& A, const Tensor& x, Tensor& y,
                               Core::Transpose trans, BLASScalar alpha, BLASScalar beta)
    {
        Dispatch::require_flag(trans, K_OP, "trans");
        Dispatch::require_dtype(x, A.values.dtype, K_OP, "x");
        Dispatch::require_dtype(y, A.values.dtype, K_OP, "y");
        Dispatch::require_real_scalar(alpha, A.values.dtype, K_OP, "alpha");
        Dispatch::require_real_scalar(beta, A.values.dtype, K_OP, "beta");

        const Dispatch::SparseMatrixOperand a = Dispatch::describe_sparse_matrix(A, K_OP, "A");
        const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, K_OP, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, K_OP, "y");

        const bool transposed = Core::is_transposed(trans);
        Dispatch::require_vector_length(vx, transposed ? a.rows : a.cols, "the columns of op(A)",
                                        K_OP, "x");
        Dispatch::require_vector_length(vy, transposed ? a.cols : a.rows, "the rows of op(A)",
                                        K_OP, "y");

        if (A.row_offsets.dtype == Core::DataType::Int32)
            return record_spmv<int32_t>(recorder_, A, a, x, vx.increment, y, vy.increment, trans,
                                        alpha, beta);
        return record_spmv<int64_t>(recorder_, A, a, x, vx.increment, y, vy.increment, trans, alpha,
                                    beta);
    }
} // namespace SushiBLAS
