/**************************************************************************/
/* sctr.cpp                                                               */
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
 * @file sctr.cpp
 * @brief Defines SparseOps::sctr, which records the sparse scatter task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/sparse/sparse.hpp>

#include <vector>
#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/sparse_kernels/sparse.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/sparse/sparse_vector_operand.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Names the operation in its messages and in its task. */
        constexpr const char* K_OP = "sparse.sctr";

        /**
         * @brief Records the scatter task of a sparse vector whose positions have type Index.
         * @tparam Index int32_t or int64_t.
         * @pre The operands have passed the checks of SparseOps::sctr.
         */
        template<typename Index>
        TaskHandle record_sctr(TaskRecorder& recorder, const SparseVector& x,
                               const Dispatch::SparseVectorOperand& sx, Tensor& y, int64_t incy)
        {
            // y is declared read too: the elements outside x's positions keep their values.
            OperandSet operands;
            operands.reads(x.indices).reads(x.values).updates(y);

            const Internal::BLASMetadata metadata = Internal::BLASMetadata(K_OP, "sparse.sctr"_op);

            const int64_t count = sx.nnz;
            const int64_t extent = sx.size;
            const int64_t first_index = sx.first_index;
            const void* read_indices = x.indices.data_or_null();
            const void* read_x = x.values.data_or_null();
            void* rw_y = y.data_or_null();

            return Internal::execute_typed_task<Dispatch::Domain::All>(
                recorder, metadata.get(), operands, x.values.dtype, "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    const Kernels::IndexList<Index> positions{
                        count, extent, first_index, static_cast<const Index*>(read_indices)};
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<T*>(rw_y);

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::sctr<T, Index>(px, positions, py, incy);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::sctr<T, Index>(q2, px, positions, py, incy,
                                                                 deps2);
                        });
                });
        }
    } // namespace

    TaskHandle SparseOps::sctr(const SparseVector& x, Tensor& y)
    {
        Dispatch::require_dtype(y, x.values.dtype, K_OP, "y");

        const Dispatch::SparseVectorOperand sx = Dispatch::describe_sparse_vector(x, K_OP, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, K_OP, "y");
        Dispatch::require_vector_length(vy, sx.size, "the size of x", K_OP, "y");

        if (x.indices.dtype == Core::DataType::Int32)
            return record_sctr<int32_t>(recorder_, x, sx, y, vy.increment);
        return record_sctr<int64_t>(recorder_, x, sx, y, vy.increment);
    }
} // namespace SushiBLAS
