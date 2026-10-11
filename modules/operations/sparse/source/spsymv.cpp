/**************************************************************************/
/* spsymv.cpp                                                             */
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
 * @file spsymv.cpp
 * @brief Defines SparseOps::spsymv and sphemv, which record the one-triangle product tasks.
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
#include <SushiBLAS/sparse/accumulator_size.hpp>
#include <SushiBLAS/sparse/sparse_matrix_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records one one-triangle product task for a CSR matrix with indices of type Index.
         * @tparam Index int32_t or int64_t.
         * @tparam Conj Kernels::Unconjugated for SPSYMV, Kernels::Conjugated for SPHEMV.
         * @pre The operands have passed the checks of record_spsymv.
         */
        template<typename Index, typename Conj>
        TaskHandle record_spsymv_task(TaskRecorder& recorder, const char* op,
                                      SushiRuntime::Graph::OpID op_id, const SparseMatrix& A,
                                      const Dispatch::SparseMatrixOperand& a, const Tensor& x,
                                      int64_t incx, Tensor& y, int64_t incy, Core::Uplo uplo,
                                      BLASScalar alpha, BLASScalar beta)
        {
            OperandSet operands;
            operands.reads(A.row_offsets).reads(A.column_indices).reads(A.values).reads(x);
            if (beta.is_zero())
                operands.writes(y);
            else
                operands.updates(y);

            // Every stored entry scatters to its mirror, so the product keeps partial sums.
            const std::size_t blocks = Kernels::SYCL::scatter_blocks(
                Device::profile_for(A.row_offsets.get_device()), a.rows, a.nnz);
            const std::size_t partial_bytes = blocks * static_cast<std::size_t>(a.rows) *
                                              Internal::accumulator_bytes(A.values.dtype);
            const ScratchBuffer scratch =
                Internal::allocate_blas_scratch(recorder, A.row_offsets, partial_bytes, operands);

            const Internal::BLASMetadata meta =
                Internal::BLASMetadata(op, op_id).scalar(alpha).scalar(beta).flag(uplo);

            const int64_t rows = a.rows;
            const int64_t cols = a.cols;
            const int64_t nnz = a.nnz;
            const int64_t first_index = a.first_index;
            const void* read_offsets = A.row_offsets.data_or_null();
            const void* read_columns = A.column_indices.data_or_null();
            const void* read_values = A.values.data_or_null();
            const void* read_x = x.data_or_null();
            void* rw_y = y.data_or_null();

            return Internal::execute_typed_task(recorder, meta.get(), operands, A.values.dtype,
                                                "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
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
                            Kernels::Reference::spsymv<T, Index, Policy>(uplo, pattern, pv, alpha_t,
                                                                         px, incx, beta_t, py,
                                                                         incy);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::spsymv<T, Index, Policy>(q2, uplo, pattern, pv,
                                                                           alpha_t, px, incx,
                                                                           beta_t, py, incy,
                                                                           partials, blocks, deps2);
                        });
                });
        }

        /**
         * @brief Checks the operands of a one-triangle product and records its task.
         * @tparam Conj Kernels::Unconjugated for SPSYMV, Kernels::Conjugated for SPHEMV.
         */
        template<typename Conj>
        TaskHandle record_spsymv(TaskRecorder& recorder, const char* op,
                                 SushiRuntime::Graph::OpID op_id, const SparseMatrix& A,
                                 const Tensor& x, Tensor& y, Core::Uplo uplo, BLASScalar alpha,
                                 BLASScalar beta)
        {
            Dispatch::require_flag(uplo, op, "uplo");
            Dispatch::require_dtype(x, A.values.dtype, op, "x");
            Dispatch::require_dtype(y, A.values.dtype, op, "y");
            Dispatch::require_real_scalar(alpha, A.values.dtype, op, "alpha");
            Dispatch::require_real_scalar(beta, A.values.dtype, op, "beta");

            const Dispatch::SparseMatrixOperand a = Dispatch::describe_sparse_matrix(A, op, "A");
            Dispatch::require_operand(a.rows == a.cols, op, "A",
                                      "is {} x {}; a symmetric or Hermitian matrix is square",
                                      a.rows, a.cols);

            const Dispatch::VectorOperand vx = Dispatch::describe_vector_operand(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_vector_length(vx, a.rows, "the order of A", op, "x");
            Dispatch::require_vector_length(vy, a.rows, "the order of A", op, "y");

            if (A.row_offsets.dtype == Core::DataType::Int32)
                return record_spsymv_task<int32_t, Conj>(recorder, op, op_id, A, a, x, vx.increment,
                                                         y, vy.increment, uplo, alpha, beta);
            return record_spsymv_task<int64_t, Conj>(recorder, op, op_id, A, a, x, vx.increment, y,
                                                     vy.increment, uplo, alpha, beta);
        }
    } // namespace

    TaskHandle SparseOps::spsymv(const SparseMatrix& A, const Tensor& x, Tensor& y,
                                 Core::Uplo uplo, BLASScalar alpha, BLASScalar beta)
    {
        return record_spsymv<Kernels::Unconjugated>(recorder_, "sparse.spsymv", "sparse.spsymv"_op,
                                                    A, x, y, uplo, alpha, beta);
    }

    TaskHandle SparseOps::sphemv(const SparseMatrix& A, const Tensor& x, Tensor& y,
                                 Core::Uplo uplo, BLASScalar alpha, BLASScalar beta)
    {
        if (!Core::is_complex_dtype(A.values.dtype))
            return spsymv(A, x, y, uplo, alpha, beta);

        return record_spsymv<Kernels::Conjugated>(recorder_, "sparse.sphemv", "sparse.sphemv"_op, A,
                                                  x, y, uplo, alpha, beta);
    }
} // namespace SushiBLAS
