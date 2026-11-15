/**************************************************************************/
/* positioned_dispatch.hpp                                                */
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
 * @file positioned_dispatch.hpp
 * @brief Declares the executor of a random fill that reads its stream position when it runs.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <type_traits>
#include <initializer_list>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/random/dispatch.hpp>
#include <SushiBLAS/random/rng_position_tensor.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /**
             * @brief Runs one positioned random-fill task body for the element type of a with_element_type arm.
             * @tparam Func The task body: `(T, const RandomFill&, const std::uint64_t*, size, T*) -> sycl::event`.
             */
            template <typename Func>
            struct PositionedRandomArm
            {
                /** @brief Holds the task body the arm calls. */
                const Func& task_func;
                /** @brief Holds the launcher the body fills through. */
                const RandomFill& fill;
                /** @brief Holds the address of the seed, followed by the offset, that the fill reads. */
                const std::uint64_t* position;
                /** @brief Holds the element count of the tensor. */
                int64_t size;
                /** @brief Holds the tensor's base address. */
                void* ptr;
                /** @brief Holds the operation name for messages. */
                const char* name;
                /** @brief Receives the event the task body returns. */
                sycl::event& done;

                /**
                 * @brief Runs the fill for the element type named by the tag.
                 * @throws std::runtime_error for HALF.
                 */
                template <typename Tag>
                void operator()(Tag) const
                {
                    using T = typename Tag::type;
                    if constexpr (std::is_same_v<T, sycl::half>)
                        SB_THROW_IF(true, "RNG operation '{}' does not support dtype HALF.", name);
                    else
                        done = task_func(T{}, fill, position, size, static_cast<T*>(ptr));
                }
            };
        } // namespace Detail

        /**
         * @brief Records a random fill that reads its seed and offset from @p position when run.
         * @tparam Dom The element families the distribution draws; HALF is always refused.
         * @param position Rank-1 INT64 of K_RNG_POSITION_ELEMENTS elements; must not overlap @p t.
         * @param params The distribution's parameters in slots 0..n-1; the seed and offset hold 0.
         * @throws std::invalid_argument on an operand violation; UnsupportedExecution on a device-only t under ForceReference.
         * @see modules/operations/random/README.md
         */
        template<Domain Dom, typename Func>
        TaskHandle execute_random_at(TaskRecorder& recorder, const Tensor& position, Tensor& t,
                                     const char* name, SushiRuntime::OperationID op_id,
                                     std::initializer_list<double> params, Func&& task_func)
        {
            require_random_dtype(t, Dom, Sampling::DrawsValues, name);
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_contiguous(t, name, "t");
            require_rng_position(position, name, "position");
            require_operand(!position.region().overlaps(t.region()), name, "position",
                            "overlaps the operand 't' it is read beside");

            const int64_t size = t.num_elements;
            void* ptr = t.data_or_null();
            const std::uint64_t* words = position.data_as<std::uint64_t>();

            OperandSet operands;
            operands.reads(position).writes(t);
            recorder.require_host_reachable(operands, "RNG operation", name);

            TaskDescription description;
            description.name = name;
            description.operation = op_id;
            set_task_params(description, params, name,
                            TaskDescription::K_MAX_PARAMETERS - K_MAX_RNG_PARAMS);
            set_rng_params(description, 0, 0);

            return recorder.record_pair(description, operands, "RNG operation",
                [dtype=t.dtype, size, words, task_func, pT = ptr, name]
                (const KernelPairDispatcher& pair,
                 const std::vector<sycl::event>& deps) -> sycl::event
                {
                    sycl::event done;
                    const RandomFill fill(pair, deps);
                    const Detail::PositionedRandomArm<std::remove_reference_t<Func>> arm{
                        task_func, fill, words, size, pT, name, done};
                    with_element_type<Dom>(dtype, "RNG operation", name, arm);
                    return done;
                });
        }
    } // namespace Dispatch
} // namespace SushiBLAS
