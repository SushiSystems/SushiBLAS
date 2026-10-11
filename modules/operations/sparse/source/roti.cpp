/**************************************************************************/
/* roti.cpp                                                               */
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
 * @file roti.cpp
 * @brief Defines SparseOps::roti, which records the sparse plane rotation task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/sparse/sparse.hpp>

#include <vector>
#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/sparse_kernels/sparse.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiBLAS/operands/task_metadata.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/sparse/sparse_vector_operand.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /** @brief Names the operation in its messages and in its task. */
        constexpr const char* K_OP = "sparse.roti";

        /**
         * @brief Records the rotation task of a sparse vector whose positions have type Index.
         * @tparam Index int32_t or int64_t.
         * @pre The operands have passed the checks of SparseOps::roti.
         */
        template<typename Index>
        TaskHandle record_roti(TaskRecorder& recorder, SparseVector& x,
                               const Dispatch::SparseVectorOperand& sx, Tensor& y, int64_t incy,
                               BLASScalar c, BLASScalar s)
        {
            OperandSet operands;
            operands.reads(x.indices).updates(x.values).updates(y);

            const Internal::BLASMetadata meta =
                Internal::BLASMetadata(K_OP, "sparse.roti"_op).scalar(c).scalar(s);

            const int64_t count = sx.nnz;
            const int64_t extent = sx.size;
            const int64_t first_index = sx.first_index;
            const void* read_indices = x.indices.data_or_null();
            void* rw_x = x.values.data_or_null();
            void* rw_y = y.data_or_null();

            return Internal::execute_typed_task(recorder, meta.get(), operands, x.values.dtype,
                                                "Sparse operation",
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);
                    using R = Kernels::RealType<T>;
                    const Kernels::IndexList<Index> positions{
                        count, extent, first_index, static_cast<const Index*>(read_indices)};
                    auto px = static_cast<T*>(rw_x);
                    auto py = static_cast<T*>(rw_y);
                    const R c_t = c.as<R>();
                    const R s_t = s.as<R>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::roti<T, Index>(px, positions, py, incy, c_t, s_t);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::roti<T, Index>(q2, px, positions, py, incy, c_t,
                                                                 s_t, deps2);
                        });
                });
        }
    } // namespace

    TaskHandle SparseOps::roti(SparseVector& x, Tensor& y, double c, double s)
    {
        Dispatch::require_dtype(y, x.values.dtype, K_OP, "y");

        const Dispatch::SparseVectorOperand sx = Dispatch::describe_sparse_vector(x, K_OP, "x");
        const Dispatch::VectorOperand vy = Dispatch::describe_vector_operand(y, K_OP, "y");
        Dispatch::require_vector_length(vy, sx.size, "the size of x", K_OP, "y");

        if (x.indices.dtype == Core::DataType::Int32)
            return record_roti<int32_t>(recorder_, x, sx, y, vy.increment, c, s);
        return record_roti<int64_t>(recorder_, x, sx, y, vy.increment, c, s);
    }
} // namespace SushiBLAS
