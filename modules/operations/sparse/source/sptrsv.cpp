/**************************************************************************/
/* sptrsv.cpp                                                             */
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
 * @file sptrsv.cpp
 * @brief Defines SparseOps::sptrsv, which records the sparse triangular solve task.
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
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/sparse/sparse_matrix_operand.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Names the operation in its messages and in its task. */
        constexpr const char* K_OP = "sparse.sptrsv";

        /**
         * @brief Records the SPTRSV task of a CSR matrix whose indices have type Index.
         * @tparam Index int32_t or int64_t.
         * @pre The operands have passed the checks of SparseOps::sptrsv.
         */
        template<typename Index>
        TaskHandle record_sptrsv(TaskRecorder& recorder, const SparseMatrix& A,
                                 const Dispatch::SparseMatrixOperand& a, Tensor& b, Core::Uplo uplo,
                                 Core::Transpose trans, Core::Diag diag)
        {
            OperandSet operands;
            operands.reads(A.row_offsets).reads(A.column_indices).reads(A.values).updates(b);

            const ScratchBuffer work = Internal::allocate_blas_scratch(
                recorder, A.row_offsets,
                Internal::running_solution_bytes(A.values.dtype) * static_cast<std::size_t>(a.rows),
                operands);

            const Internal::BlasMetadata meta = Internal::BlasMetadata(K_OP, "sparse.sptrsv"_op)
                                                    .flag(uplo).flag(trans).flag(diag);

            const int64_t rows = a.rows;
            const int64_t cols = a.cols;
            const int64_t nnz = a.nnz;
            const int64_t first_index = a.first_index;
            const void* read_offsets = A.row_offsets.data_or_null();
            const void* read_columns = A.column_indices.data_or_null();
            const void* read_values = A.values.data_or_null();
            void* rw_b = b.data_or_null();

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
                    auto pb = static_cast<T*>(rw_b);
                    auto wide = static_cast<Kernels::Accumulator<T>*>(work.data());

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::sptrsv<T, Index>(uplo, trans, diag, pattern, pv,
                                                                 pb);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::sptrsv<T, Index>(q2, uplo, trans, diag, pattern,
                                                                   pv, pb, wide, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle SparseOps::sptrsv(const SparseMatrix& A, Tensor& b, Core::Uplo uplo,
                                 Core::Transpose trans, Core::Diag diag)
    {
        Dispatch::require_flag(uplo, K_OP, "uplo");
        Dispatch::require_flag(trans, K_OP, "trans");
        Dispatch::require_flag(diag, K_OP, "diag");
        Dispatch::require_dtype(b, A.values.dtype, K_OP, "b");

        const Dispatch::SparseMatrixOperand a = Dispatch::describe_sparse_matrix(A, K_OP, "A");
        Dispatch::require_operand(a.rows == a.cols, K_OP, "A",
                                  "is {} x {}; a triangular solve needs a square matrix", a.rows,
                                  a.cols);

        const Dispatch::VectorOperand vb = Dispatch::describe_vector_operand(b, K_OP, "b");
        Dispatch::require_unit_increment(vb, K_OP, "b");
        Dispatch::require_vector_length(vb, a.rows, "the order of A", K_OP, "b");

        if (A.row_offsets.dtype == Core::DataType::Int32)
            return record_sptrsv<int32_t>(recorder_, A, a, b, uplo, trans, diag);
        return record_sptrsv<int64_t>(recorder_, A, a, b, uplo, trans, diag);
    }
} // namespace SushiBLAS
