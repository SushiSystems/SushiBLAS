/**************************************************************************/
/* dispatch.hpp                                                           */
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
 * @file dispatch.hpp
 * @brief Declares the broadcast executor of the dispatch seam.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <initializer_list>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/functor_map/map_operand.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiBLAS/functor_map/broadcast_task.hpp>
#include <SushiBLAS/broadcast/broadcast_checks.hpp>
#include <SushiBLAS/broadcast/broadcast_operand.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /** @brief Holds the most vectors one broadcast call takes. */
        constexpr std::size_t K_MAX_BROADCAST_VECTORS = 4;

        /**
         * @brief Records `C[o, n] = op(A[o, n], v0, ..., vN-1)`, or nothing when A is empty.
         * @param C May be A itself; any other overlap with A, and any with a vector, is refused.
         * @param op_func Callable `(auto x, auto v0, ..., auto vN-1) -> auto`, or a bind_scalars
         *        or bind_reads bundle of one.
         * @throws std::invalid_argument on an operand, declared-read or parameter violation.
         * @see modules/operations/broadcast/README.md
         */
        template<Domain Dom = Domain::Real, std::size_t N, typename Func, typename Param = double>
        TaskHandle execute_broadcast(TaskRecorder& recorder, const Tensor& A,
                                     const std::array<BroadcastOperand, N>& vectors, Tensor& C,
                                     const char* name, SushiRuntime::OperationID op_id,
                                     Func&& op_func, std::initializer_list<Param> params = {})
        {
            static_assert(N >= 1 && N <= K_MAX_BROADCAST_VECTORS,
                          "A broadcast call takes one to four vectors.");
            static_assert(!Detail::IsWriteBound<std::decay_t<Func>>::value,
                          "execute_broadcast takes no bind_writes bundle: its output is the "
                          "write it declares; a functor that writes more records "
                          "through IndexOps::for_each.");

            constexpr bool k_binds = !Detail::DeclaresReads<std::decay_t<Func>>::value;
            Detail::require_map_storage<k_binds>(A, name, "A");
            require_device_dtype(A, name, "A");
            Detail::require_broadcast_layout(A, name, "A");
            require_contiguous(A, name, "A");
            Detail::require_map_storage<k_binds>(C, name, "C");
            require_device_dtype(C, name, "C");
            Detail::require_broadcast_layout(C, name, "C");
            require_contiguous(C, name, "C");
            require_same_shape(A, C, name, "A", "C");
            require_dtype(C, A.dtype, name, "C");

            require_domain<Dom>(A.dtype, "Broadcast operation", name);
            Detail::require_declared_reads(op_func, A.dtype, name);

            if (A.num_elements == 0)
                return TaskHandle();

            Detail::require_broadcast_output(A, C, name);
            const Detail::BroadcastGeometry geometry =
                Detail::require_broadcast_operands<k_binds>(A, vectors.data(), N, C, name);

            OperandSet operands;
            operands.reads(A);
            for (std::size_t k = 0; k < N; ++k)
                operands.reads(*vectors[k].vector);
            operands.writes(C);
            Detail::declare_reads(op_func, operands, name);

            Detail::BroadcastBinding<N> binding{geometry.outer,
                                                Detail::map_operand<k_binds>(operands, C),
                                                Detail::map_operand<k_binds>(operands, A),
                                                {},
                                                geometry.inner,
                                                {}};
            for (std::size_t k = 0; k < N; ++k)
            {
                binding.vectors[k] = Detail::map_operand<k_binds>(operands, *vectors[k].vector);
                binding.selected_by[k] = vectors[k].axis == BroadcastAxis::Leading
                                             ? BroadcastIndex::Outer
                                             : BroadcastIndex::Inner;
            }

            TaskDescription description;
            description.name = name;
            description.operation = op_id;
            description.shape = ElementwiseShape{C.num_elements, A.dtype};
            set_task_params(description, params, name);

            return recorder.record_pair(description, operands, "Broadcast operation",
                Detail::make_operand_broadcast_task<Dom>(A.dtype, "Broadcast operation", name,
                                                         binding, Detail::task_functor(op_func)));
        }
    } // namespace Dispatch
} // namespace SushiBLAS
