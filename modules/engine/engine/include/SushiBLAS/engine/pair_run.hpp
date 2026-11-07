/**************************************************************************/
/* pair_run.hpp                                                           */
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


/**
 * @file pair_run.hpp
 * @brief Declares PairRun, the one call of a recorded kernel-pair body: its dispatcher, its
 *        dependencies and the event that ends its work.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds what one run of a recorded body gets and what it hands back.
     *
     * A sink builds one per run, passes it to the body and reads finished() after. It keeps the
     * SYCL types out of the TaskSink port, which names PairRun only by reference.
     * @see modules/engine/engine/README.md
     */
    class PairRun final
    {
        public:
            /**
             * @brief Binds the run to a dispatcher and the events it is ordered behind.
             * @param dispatcher Must outlive the run.
             * @param dependencies Must outlive the run.
             */
            PairRun(const KernelPairDispatcher& dispatcher,
                    const std::vector<sycl::event>& dependencies) noexcept
                : dispatcher_(dispatcher), dependencies_(dependencies)
            {

            }

            /** @brief Returns the dispatcher the body routes its kernel pair through. */
            [[nodiscard]] const KernelPairDispatcher& dispatcher() const noexcept { return dispatcher_; }

            /** @brief Returns the events the body's work is ordered behind. */
            [[nodiscard]] const std::vector<sycl::event>& dependencies() const noexcept
            {
                return dependencies_;
            }

            /** @brief Records the event that completes after every kernel the body submitted. */
            void finish(sycl::event done) noexcept { done_ = std::move(done); }

            /** @brief Returns the event finish() recorded, or an empty one when the body submitted nothing. */
            [[nodiscard]] const sycl::event& finished() const noexcept { return done_; }

        private:
            /// @brief The dispatcher of this run.
            const KernelPairDispatcher& dispatcher_;

            /// @brief The events this run is ordered behind.
            const std::vector<sycl::event>& dependencies_;

            /// @brief The event that ends the body's work.
            sycl::event done_;
    };
} // namespace SushiBLAS
