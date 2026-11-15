/**************************************************************************/
/* take_along.cpp                                                         */
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
 * @file take_along.cpp
 * @brief Defines RowIndexOps::take_along, which records the pick of one element per row.
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
        /** @brief Holds what the take_along task needs when it runs. */
        struct TakeAlongOperands
        {
            Kernels::MatrixSpan values_span; /**< Extents and strides of the values. */
            int64_t index_stride;            /**< Elements between two indices. */
            int64_t picked_stride;           /**< Elements between two picked elements. */
            Core::DataType index_dtype;      /**< INT32 or INT64. */
            const void* values;              /**< The first element of the values. */
            const void* index;               /**< The first index. */
            void* picked;                    /**< The first picked element. */
        };

        /** @brief Runs the reference or the SYCL kernel for one element and index type. */
        template<typename T, typename Index>
        sycl::event run_take_along(const TakeAlongOperands& op,
                                   const KernelPairDispatcher& dispatcher,
                                   const std::vector<sycl::event>& deps)
        {
            const auto values = static_cast<const T*>(op.values);
            const auto positions = static_cast<const Index*>(op.index);
            const auto picked = static_cast<T*>(op.picked);

            return dispatcher.run(deps,
                [=]
                {
                    Kernels::Reference::take_along<T, Index>(op.values_span, values, positions,
                                                             op.index_stride, picked,
                                                             op.picked_stride);
                },
                [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                {
                    return Kernels::SYCL::take_along<T, Index>(q2, op.values_span, values,
                                                               positions, op.index_stride, picked,
                                                               op.picked_stride, deps2);
                });
        }
    } // namespace

    TaskHandle RowIndexOps::take_along(const Tensor& values, const Tensor& index, Tensor& picked)
    {
        static constexpr const char* NAME = "index.take_along";

        Gather::require_data_operand(values, 2, NAME, "values");
        Gather::require_data_operand(picked, 1, NAME, "picked");
        Gather::require_index_operand(index, NAME, "index");
        Dispatch::require_dtype(picked, values.dtype, NAME, "picked");
        Dispatch::require_extent(index.shape[0], values.shape[0], "positions", NAME, "index");
        Dispatch::require_extent(picked.shape[0], values.shape[0], "elements", NAME, "picked");
        Gather::require_distinct_elements(picked, NAME, "picked");
        Gather::require_disjoint(picked, values, NAME, "picked", "values");
        Gather::require_disjoint(picked, index, NAME, "picked", "index");

        const TakeAlongOperands op{
            {values.shape[0], values.shape[1], values.strides[0], values.strides[1]},
            index.strides[0], picked.strides[0], index.dtype,
            values.data_or_null(), index.data_or_null(), picked.data_or_null()};

        OperandSet operands;
        operands.reads(values).reads(index).writes(picked);

        TaskDescription description;
        description.name = NAME;
        description.operation = "index.take_along"_op;
        description.shape = ElementwiseShape{picked.num_elements, picked.dtype};

        return Internal::execute_typed_task<Dispatch::Domain::All>(
            recorder_, description, operands, values.dtype, "Row index operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if (op.index_dtype == Core::DataType::Int64)
                    return run_take_along<T, int64_t>(op, dispatcher, deps);

                return run_take_along<T, int32_t>(op, dispatcher, deps);
            });
    }
} // namespace SushiBLAS
