/**************************************************************************/
/* gthr.cpp                                                               */
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
 * @file gthr.cpp
 * @brief Defines SparseOps::gthr, which records the sparse gather task.
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
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/sparse/sparse_vector_operand.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Names the operation in its messages and in its task. */
        constexpr const char* K_OP = "sparse.gthr";

        /**
         * @brief Records the gather task of a sparse vector whose positions have type Index.
         * @tparam Index int32_t or int64_t.
         * @pre The operands have passed the checks of SparseOps::gthr.
         */
        template<typename Index>
        TaskHandle record_gthr(TaskRecorder& recorder, const Tensor& y, int64_t incy,
                               SparseVector& x, const Dispatch::SparseVectorOperand& sx)
        {
            OperandSet operands;
            operands.reads(x.indices).reads(y).writes(x.values);

            const Internal::BLASMetadata meta = Internal::BLASMetadata(K_OP, "sparse.gthr"_op);

            const int64_t count = sx.nnz;
            const int64_t extent = sx.size;
            const int64_t first_index = sx.first_index;
            const void* read_indices = x.indices.data_or_null();
            const void* read_y = y.data_or_null();
            void* write_x = x.values.data_or_null();

            return Internal::execute_typed_task<Dispatch::Domain::All>(
                recorder, meta.get(), operands, x.values.dtype, "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    const Kernels::IndexList<Index> positions{
                        count, extent, first_index, static_cast<const Index*>(read_indices)};
                    auto py = static_cast<const T*>(read_y);
                    auto px = static_cast<T*>(write_x);

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::gthr<T, Index>(py, incy, positions, px);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::gthr<T, Index>(q2, py, incy, positions, px,
                                                                 deps2);
                        });
                });
        }
    } // namespace

    TaskHandle SparseOps::gthr(const Tensor& y, SparseVector& x)
    {
        Dispatch::require_dtype(y, x.values.dtype, K_OP, "y");

        const Dispatch::SparseVectorOperand sx = Dispatch::describe_sparse_vector(x, K_OP, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, K_OP, "y");
        Dispatch::require_vector_length(vy, sx.size, "the size of x", K_OP, "y");

        if (x.indices.dtype == Core::DataType::Int32)
            return record_gthr<int32_t>(recorder_, y, vy.increment, x, sx);
        return record_gthr<int64_t>(recorder_, y, vy.increment, x, sx);
    }
} // namespace SushiBLAS
