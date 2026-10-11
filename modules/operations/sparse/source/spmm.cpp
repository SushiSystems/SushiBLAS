/**************************************************************************/
/* spmm.cpp                                                               */
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
 * @file spmm.cpp
 * @brief Defines SparseOps::spmm, which records the sparse-times-dense matrix product task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/sparse/sparse.hpp>

#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/sparse_kernels/sparse.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/sparse/sparse_matrix_operand.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Names the operation in its messages and in its task. */
        constexpr const char* K_OP = "sparse.spmm";

        /**
         * @brief Records the SPMM task of a CSR matrix whose indices have type Index.
         * @tparam Index int32_t or int64_t.
         * @pre The operands have passed the checks of SparseOps::spmm.
         */
        template<typename Index>
        TaskHandle record_spmm(TaskRecorder& recorder, const SparseMatrix& A,
                               const Dispatch::SparseMatrixOperand& a, const Tensor& B,
                               const Dispatch::MatrixOperand& b, Tensor& C,
                               const Dispatch::MatrixOperand& c, Core::Transpose trans,
                               BlasScalar alpha, BlasScalar beta)
        {
            OperandSet operands;
            operands.reads(A.row_offsets).reads(A.column_indices).reads(A.values).reads(B);
            if (beta.is_zero())
                operands.writes(C);
            else
                operands.updates(C);

            // Only a transposed product carries C unrounded, and only where the dtype needs it.
            const std::size_t work_elements = Core::is_transposed(trans)
                ? static_cast<std::size_t>(c.matrix_elements)
                : std::size_t{0};
            const ScratchBuffer work = Internal::allocate_blas_scratch(
                recorder, A.row_offsets,
                Internal::running_solution_bytes(A.values.dtype) * work_elements, operands);

            const Internal::BlasMetadata meta = Internal::BlasMetadata(K_OP, "sparse.spmm"_op)
                                                    .scalar(alpha).scalar(beta).flag(trans);

            const Core::Layout layout = c.layout;
            const int64_t rows = a.rows;
            const int64_t cols = a.cols;
            const int64_t nnz = a.nnz;
            const int64_t first_index = a.first_index;
            const int64_t nrhs = c.cols;
            const int64_t ldb = b.leading_dimension;
            const int64_t ldc = c.leading_dimension;
            const void* read_offsets = A.row_offsets.data_or_null();
            const void* read_columns = A.column_indices.data_or_null();
            const void* read_values = A.values.data_or_null();
            const void* read_B = B.data_or_null();
            void* rw_C = C.data_or_null();

            return Internal::execute_typed_task(recorder, meta.get(), operands, A.values.dtype,
                                                "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    const Kernels::CsrPattern<Index> pattern{
                        rows, cols, nnz, first_index, static_cast<const Index*>(read_offsets),
                        static_cast<const Index*>(read_columns)};
                    auto pv = static_cast<const T*>(read_values);
                    auto pB = static_cast<const T*>(read_B);
                    auto pC = static_cast<T*>(rw_C);
                    const T alpha_t = alpha.as<T>();
                    const T beta_t = beta.as<T>();
                    auto wide = static_cast<Kernels::Accumulator<T>*>(work.data());

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::spmm<T, Index>(layout, trans, pattern, pv, nrhs,
                                                               alpha_t, pB, ldb, beta_t, pC, ldc);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::spmm<T, Index>(q2, layout, trans, pattern, pv,
                                                                 nrhs, alpha_t, pB, ldb, beta_t, pC,
                                                                 ldc, wide, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle SparseOps::spmm(const SparseMatrix& A, const Tensor& B, Tensor& C,
                               Core::Transpose trans, BlasScalar alpha, BlasScalar beta)
    {
        Dispatch::require_flag(trans, K_OP, "trans");
        Dispatch::require_dtype(B, A.values.dtype, K_OP, "B");
        Dispatch::require_dtype(C, A.values.dtype, K_OP, "C");
        Dispatch::require_real_scalar(alpha, A.values.dtype, K_OP, "alpha");
        Dispatch::require_real_scalar(beta, A.values.dtype, K_OP, "beta");

        const Dispatch::SparseMatrixOperand a = Dispatch::describe_sparse_matrix(A, K_OP, "A");
        Dispatch::require_rank(B, 2, K_OP, "B");
        Dispatch::require_rank(C, 2, K_OP, "C");
        const Dispatch::MatrixOperand b = Dispatch::describe_matrix_operand(B, K_OP, "B");
        const Dispatch::MatrixOperand c = Dispatch::describe_matrix_operand(C, K_OP, "C");
        Dispatch::require_same_layout(b, c, K_OP, "B", "C");

        const bool transposed = Core::is_transposed(trans);
        Dispatch::require_matrix_extent(b.rows, transposed ? a.rows : a.cols,
                                        "rows against the columns of op(A)", K_OP, "B");
        Dispatch::require_matrix_extent(c.rows, transposed ? a.cols : a.rows,
                                        "rows against the rows of op(A)", K_OP, "C");
        Dispatch::require_matrix_extent(c.cols, b.cols, "columns against the columns of B", K_OP,
                                        "C");

        if (A.row_offsets.dtype == Core::DataType::Int32)
            return record_spmm<int32_t>(recorder_, A, a, B, b, C, c, trans, alpha, beta);
        return record_spmm<int64_t>(recorder_, A, a, B, b, C, c, trans, alpha, beta);
    }
} // namespace SushiBLAS
