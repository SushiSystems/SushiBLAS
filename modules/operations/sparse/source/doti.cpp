/**************************************************************************/
/* doti.cpp                                                               */
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
 * @file doti.cpp
 * @brief Defines SparseOps::doti, dotui and dotci, which record the sparse dot product tasks.
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
#include <SushiBLAS/operands/scalar_result.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/sparse/sparse_vector_operand.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Records one sparse dot product task for positions of type Index.
         * @tparam Index int32_t or int64_t.
         * @tparam Conj Kernels::Unconjugated for DOTI and DOTUI, Kernels::Conjugated for DOTCI.
         * @pre The operands have passed the checks of record_doti.
         */
        template<typename Index, typename Conj>
        TaskHandle record_doti_task(TaskRecorder& recorder, const char* op,
                                    SushiRuntime::Graph::OpID op_id, const SparseVector& x,
                                    const Dispatch::SparseVectorOperand& sx, const Tensor& y,
                                    int64_t incy, Tensor& result)
        {
            OperandSet operands;
            operands.reads(x.indices).reads(x.values).reads(y).writes(result);

            const ScratchBuffer scratch = Internal::allocate_level1_scratch(
                recorder, x.values, static_cast<std::size_t>(sx.nnz),
                Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, operands);

            const Internal::BLASMetadata meta = Internal::BLASMetadata(op, op_id);

            const int64_t count = sx.nnz;
            const int64_t extent = sx.size;
            const int64_t first_index = sx.first_index;
            const void* read_indices = x.indices.data_or_null();
            const void* read_x = x.values.data_or_null();
            const void* read_y = y.data_or_null();
            void* write_result = result.data_or_null();

            return Internal::execute_typed_task(recorder, meta.get(), operands, x.values.dtype,
                                                "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using Policy = Kernels::ConjugationFor<T, Conj>;
                    const Kernels::IndexList<Index> positions{
                        count, extent, first_index, static_cast<const Index*>(read_indices)};
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<const T*>(read_y);
                    auto pr = static_cast<T*>(write_result);

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::doti<T, Index, Policy>(px, positions, py, incy, pr);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::doti<T, Index, Policy>(q2, px, positions, py,
                                                                         incy, pr, scratch.data(),
                                                                         scratch.bytes(), deps2);
                        });
                });
        }

        /**
         * @brief Checks the operands of a sparse dot product and records its task.
         * @tparam Conj Kernels::Unconjugated for DOTI and DOTUI, Kernels::Conjugated for DOTCI.
         */
        template<typename Conj>
        TaskHandle record_doti(TaskRecorder& recorder, const char* op,
                               SushiRuntime::Graph::OpID op_id, const SparseVector& x,
                               const Tensor& y, Tensor& result)
        {
            Dispatch::require_dtype(y, x.values.dtype, op, "y");

            const Dispatch::SparseVectorOperand sx = Dispatch::describe_sparse_vector(x, op, "x");
            const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, op, "y");
            Dispatch::require_vector_length(vy, sx.size, "the size of x", op, "y");
            Internal::require_scalar_result(result, x.values.dtype, op);

            if (x.indices.dtype == Core::DataType::Int32)
                return record_doti_task<int32_t, Conj>(recorder, op, op_id, x, sx, y, vy.increment,
                                                       result);
            return record_doti_task<int64_t, Conj>(recorder, op, op_id, x, sx, y, vy.increment,
                                                   result);
        }
    } // namespace

    TaskHandle SparseOps::doti(const SparseVector& x, const Tensor& y, Tensor& result)
    {
        Dispatch::require_real_dtype(x.values, "sparse.doti", "x", "dotui or dotci");
        return record_doti<Kernels::Unconjugated>(recorder_, "sparse.doti", "sparse.doti"_op, x, y,
                                                  result);
    }

    TaskHandle SparseOps::dotui(const SparseVector& x, const Tensor& y, Tensor& result)
    {
        if (!Core::is_complex_dtype(x.values.dtype))
            return doti(x, y, result);

        return record_doti<Kernels::Unconjugated>(recorder_, "sparse.dotui", "sparse.dotui"_op, x,
                                                  y, result);
    }

    TaskHandle SparseOps::dotci(const SparseVector& x, const Tensor& y, Tensor& result)
    {
        if (!Core::is_complex_dtype(x.values.dtype))
            return doti(x, y, result);

        return record_doti<Kernels::Conjugated>(recorder_, "sparse.dotci", "sparse.dotci"_op, x, y,
                                                result);
    }
} // namespace SushiBLAS
