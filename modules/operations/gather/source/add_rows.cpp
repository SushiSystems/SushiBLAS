/**************************************************************************/
/* add_rows.cpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file add_rows.cpp
 * @brief Defines RowIndexOps::add_rows, which records the sum of rows into a table by index.
 * @author Mustafa Garip
 */

#include <SushiBLAS/gather/gather.hpp>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/gather/index_operands.hpp>
#include <SushiBLAS/index_kernels/index.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Holds what the add_rows task needs when it runs. */
        struct AddRowsOperands
        {
            Kernels::MatrixSpan rows_span;  /**< Extents and strides of the rows. */
            Kernels::MatrixSpan table_span; /**< Extents and strides of the table. */
            int64_t index_stride;           /**< Elements between two indices. */
            Core::DataType index_dtype;     /**< INT32 or INT64. */
            double beta;                    /**< The factor on the table; zero does not read it. */
            const void* rows;               /**< The first element of the rows. */
            const void* index;              /**< The first index. */
            void* table;                    /**< The first element of the table. */
        };

        /**
         * @brief Runs the reference or the SYCL kernel for one element type and one index width.
         * @param work The unrounded table, or null where the element type needs none.
         */
        template<typename T, typename Index>
        sycl::event run_add_rows(const AddRowsOperands& op, void* work,
                                 const KernelPairDispatcher& dispatcher,
                                 const std::vector<sycl::event>& deps)
        {
            const auto rows = static_cast<const T*>(op.rows);
            const auto positions = static_cast<const Index*>(op.index);
            const auto table = static_cast<T*>(op.table);
            const auto wide = static_cast<Kernels::Accumulator<T>*>(work);

            return dispatcher.run(deps,
                [=]
                {
                    Kernels::Reference::add_rows<T, Index>(op.rows_span, rows, positions,
                                                           op.index_stride, op.table_span, table,
                                                           op.beta);
                },
                [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                {
                    return Kernels::SYCL::add_rows<T, Index>(q2, op.rows_span, rows, positions,
                                                             op.index_stride, op.table_span, table,
                                                             op.beta, wide, deps2);
                });
        }
    } // namespace

    TaskHandle RowIndexOps::add_rows(const Tensor& rows, const Tensor& index, Tensor& table,
                                     double beta)
    {
        static constexpr const char* NAME = "index.add_rows";
        static constexpr const char* FAMILY = "Row index operation";

        Gather::require_data_operand(rows, 2, NAME, "rows");
        Gather::require_data_operand(table, 2, NAME, "table");
        Gather::require_index_operand(index, NAME, "index");
        Dispatch::require_dtype(table, rows.dtype, NAME, "table");
        Dispatch::require_domain<Dispatch::Domain::Real>(rows.dtype, FAMILY, NAME);
        Dispatch::require_extent(index.shape[0], rows.shape[0], "positions", NAME, "index");
        Dispatch::require_extent(table.shape[1], rows.shape[1], "columns", NAME, "table");
        Gather::require_distinct_elements(table, NAME, "table");
        Gather::require_disjoint(table, rows, NAME, "table", "rows");
        Gather::require_disjoint(table, index, NAME, "table", "index");

        const AddRowsOperands op{
            {rows.shape[0], rows.shape[1], rows.strides[0], rows.strides[1]},
            {table.shape[0], table.shape[1], table.strides[0], table.strides[1]},
            index.strides[0], index.dtype, beta,
            rows.data_or_null(), index.data_or_null(), table.data_or_null()};

        // A table that is scaled is read, so the task orders itself after the last writer of it.
        OperandSet operands;
        operands.reads(rows).reads(index);
        if (beta == 0.0)
            operands.writes(table);
        else
            operands.updates(table);

        // Only a dtype that rounds a sum needs the table carried unrounded between additions.
        const ScratchBuffer work = Internal::allocate_blas_scratch(
            recorder_, table,
            Internal::running_solution_bytes(table.dtype) * Gather::spanned_elements(table),
            operands);

        TaskDescription description;
        description.name = NAME;
        description.operation = "index.add_rows"_op;
        description.shape = ElementwiseShape{rows.num_elements, rows.dtype};

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, rows.dtype, FAMILY,
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if (op.index_dtype == Core::DataType::Int64)
                    return run_add_rows<T, int64_t>(op, work.data(), dispatcher, deps);

                return run_add_rows<T, int32_t>(op, work.data(), dispatcher, deps);
            });
    }
} // namespace SushiBLAS
