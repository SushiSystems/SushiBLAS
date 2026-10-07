/**************************************************************************/
/* axpyi.cpp                                                              */
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
 * @file axpyi.cpp
 * @brief Defines SparseOps::axpyi, which records the sparse AXPYI task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/sparse.hpp>

#include <vector>
#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <kernels/sparse.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/blas/task_metadata.hpp>
#include <SushiBLAS/engine/blas/argument_check.hpp>
#include <SushiBLAS/engine/blas/vector_operand.hpp>
#include <SushiBLAS/engine/sparse/sparse_vector_operand.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Names the operation in its messages and in its task. */
        constexpr const char* K_OP = "sparse.axpyi";

        /**
         * @brief Records the AXPYI task of a sparse vector whose positions have type Index.
         * @tparam Index int32_t or int64_t.
         * @pre The operands have passed the checks of SparseOps::axpyi.
         */
        template<typename Index>
        TaskHandle record_axpyi(TaskRecorder& recorder, BlasScalar alpha, const SparseVector& x,
                                const Dispatch::SparseVectorOperand& sx, Tensor& y, int64_t incy)
        {
            OperandSet operands;
            operands.reads(x.indices).reads(x.values).updates(y);

            const Internal::BlasMetadata meta =
                Internal::BlasMetadata(K_OP, "sparse.axpyi"_op).scalar(alpha);

            const int64_t count = sx.nnz;
            const int64_t extent = sx.size;
            const int64_t first_index = sx.first_index;
            const void* read_indices = x.indices.data_or_null();
            const void* read_x = x.values.data_or_null();
            void* rw_y = y.data_or_null();

            return Internal::execute_typed_task(recorder, meta.get(), operands, x.values.dtype,
                                                "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    const Kernels::IndexList<Index> positions{
                        count, extent, first_index, static_cast<const Index*>(read_indices)};
                    auto px = static_cast<const T*>(read_x);
                    auto py = static_cast<T*>(rw_y);
                    const T alpha_t = alpha.as<T>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::axpyi<T, Index>(alpha_t, px, positions, py, incy);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::axpyi<T, Index>(q2, alpha_t, px, positions, py,
                                                                  incy, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle SparseOps::axpyi(BlasScalar alpha, const SparseVector& x, Tensor& y)
    {
        Dispatch::require_dtype(y, x.values.dtype, K_OP, "y");
        Dispatch::require_real_scalar(alpha, x.values.dtype, K_OP, "alpha");

        const Dispatch::SparseVectorOperand sx = Dispatch::describe_sparse_vector(x, K_OP, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, K_OP, "y");
        Dispatch::require_vector_length(vy, sx.size, "the size of x", K_OP, "y");

        if (x.indices.dtype == Core::DataType::INT32)
            return record_axpyi<int32_t>(recorder_, alpha, x, sx, y, vy.increment);
        return record_axpyi<int64_t>(recorder_, alpha, x, sx, y, vy.increment);
    }
} // namespace SushiBLAS
