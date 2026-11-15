/**************************************************************************/
/* take_rows.cpp                                                          */
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
 * @file take_rows.cpp
 * @brief Defines RowIndexOps::take_rows, which records the copy of table rows by index.
 * @author Mustafa Garip
 */

#include <SushiBLAS/gather/gather.hpp>

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/gather/index_operands.hpp>
#include <SushiBLAS/index_kernels/index.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Holds what the take_rows task needs when it runs. */
        struct TakeRowsOperands
        {
            Kernels::MatrixSpan table_span; /**< Extents and strides of the table. */
            Kernels::MatrixSpan rows_span;  /**< Extents and strides of the rows. */
            int64_t index_stride;           /**< Elements between two indices. */
            Core::DataType index_dtype;     /**< INT32 or INT64. */
            const void* table;              /**< The first element of the table. */
            const void* index;              /**< The first index. */
            void* rows;                     /**< The first element of the rows. */
        };

        /** @brief Runs the reference or the SYCL kernel for one element and index type. */
        template<typename T, typename Index>
        sycl::event run_take_rows(const TakeRowsOperands& op,
                                  const KernelPairDispatcher& dispatcher,
                                  const std::vector<sycl::event>& deps)
        {
            const auto table = static_cast<const T*>(op.table);
            const auto positions = static_cast<const Index*>(op.index);
            const auto rows = static_cast<T*>(op.rows);

            return dispatcher.run(deps,
                [=]
                {
                    Kernels::Reference::take_rows<T, Index>(op.table_span, table, positions,
                                                            op.index_stride, op.rows_span, rows);
                },
                [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                {
                    return Kernels::SYCL::take_rows<T, Index>(q2, op.table_span, table, positions,
                                                              op.index_stride, op.rows_span, rows,
                                                              deps2);
                });
        }
    } // namespace

    TaskHandle RowIndexOps::take_rows(const Tensor& table, const Tensor& index, Tensor& rows)
    {
        static constexpr const char* NAME = "index.take_rows";

        Gather::require_data_operand(table, 2, NAME, "table");
        Gather::require_data_operand(rows, 2, NAME, "rows");
        Gather::require_index_operand(index, NAME, "index");
        Dispatch::require_dtype(rows, table.dtype, NAME, "rows");
        Dispatch::require_extent(index.shape[0], rows.shape[0], "positions", NAME, "index");
        Dispatch::require_extent(rows.shape[1], table.shape[1], "columns", NAME, "rows");
        Gather::require_distinct_elements(rows, NAME, "rows");
        Gather::require_disjoint(rows, table, NAME, "rows", "table");
        Gather::require_disjoint(rows, index, NAME, "rows", "index");

        const TakeRowsOperands op{
            {table.shape[0], table.shape[1], table.strides[0], table.strides[1]},
            {rows.shape[0], rows.shape[1], rows.strides[0], rows.strides[1]},
            index.strides[0], index.dtype,
            table.data_or_null(), index.data_or_null(), rows.data_or_null()};

        OperandSet operands;
        operands.reads(table).reads(index).writes(rows);

        TaskDescription description;
        description.name = NAME;
        description.operation = "index.take_rows"_op;
        description.shape = ElementwiseShape{rows.num_elements, rows.dtype};

        return Internal::execute_typed_task<Dispatch::Domain::All>(
            recorder_, description, operands, table.dtype, "Row index operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if (op.index_dtype == Core::DataType::Int64)
                    return run_take_rows<T, int64_t>(op, dispatcher, deps);

                return run_take_rows<T, int32_t>(op, dispatcher, deps);
            });
    }
} // namespace SushiBLAS
