/**************************************************************************/
/* kernel_pair_dispatcher.hpp                                             */
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
 * @file kernel_pair_dispatcher.hpp
 * @brief Declares KernelPairDispatcher, which routes one kernel of a reference and SYCL pair and
 *        holds the queue the SYCL half runs on.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>

namespace SushiBLAS
{
    /**
     * @brief Routes an operation's reference or SYCL kernel by the mode fixed at record time and
     *        holds the queue the SYCL kernel is submitted to.
     *
     * A dispatcher on a host engine holds no queue, so a body reaches the queue only through
     * queue(), which refuses there.
     * @see modules/engine/engine/README.md
     */
    class KernelPairDispatcher final
    {
        public:
            /** @brief Fixes the execution mode every run() routes by and holds no queue. */
            explicit KernelPairDispatcher(ExecutionMode mode) noexcept
                : mode_(mode), queue_(nullptr)
            {

            }

            /**
             * @brief Fixes the execution mode and the queue the SYCL half runs on.
             * @param queue Must outlive the dispatcher.
             */
            KernelPairDispatcher(ExecutionMode mode, sycl::queue& queue) noexcept
                : mode_(mode), queue_(&queue)
            {

            }

            /** @brief Returns the mode every run() routes by. */
            [[nodiscard]] ExecutionMode mode() const noexcept { return mode_; }

            /** @brief Reports whether the dispatcher holds a queue. */
            [[nodiscard]] bool has_queue() const noexcept { return queue_ != nullptr; }

            /**
             * @brief Returns the queue the dispatcher holds.
             * @throws UnsupportedExecution when the dispatcher holds none, as on a host engine.
             */
            [[nodiscard]] sycl::queue& queue() const
            {
                if (queue_ == nullptr)
                {
                    throw UnsupportedExecution(
                        "The kernel pair dispatcher holds no queue: a host engine runs no SYCL "
                        "device.");
                }
                return *queue_;
            }

            /**
             * @brief Runs reference_fn on the host under ForceReference, else sycl_fn on the held queue.
             * @param reference_fn Callable `() -> void` running the reference kernel.
             * @param sycl_fn Callable `(sycl::queue&, deps)` returning a `sycl::event`, or
             *        Device::SubmittedEvents for several kernels.
             * @return What sycl_fn returns, or an empty value of its type under ForceReference.
             * @throws UnsupportedExecution outside ForceReference when no queue is held.
             */
            template<typename ReferenceFn, typename SYCLFn>
            auto run(const std::vector<sycl::event>& deps, ReferenceFn&& reference_fn,
                     SYCLFn&& sycl_fn) const
            {
                using Result = std::decay_t<
                    std::invoke_result_t<SYCLFn&&, sycl::queue&, const std::vector<sycl::event>&>>;

                if (mode_ == ExecutionMode::ForceReference)
                {
                    sycl::event::wait(deps);
                    std::forward<ReferenceFn>(reference_fn)();
                    return Result();
                }

                return std::forward<SYCLFn>(sycl_fn)(queue(), deps);
            }

        private:
            /// @brief The recorder's execution mode when the task was recorded.
            ExecutionMode mode_;

            /// @brief The queue the SYCL half runs on; null on a host engine.
            sycl::queue* queue_;
    };
} // namespace SushiBLAS
