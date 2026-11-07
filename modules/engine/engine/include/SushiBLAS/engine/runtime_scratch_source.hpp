/**************************************************************************/
/* runtime_scratch_source.hpp                                             */
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
 * @file runtime_scratch_source.hpp
 * @brief Declares RuntimeScratchSource, the ScratchBlockSource that allocates device memory from a runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/engine/scratch_block_source.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @brief Serves every block as a Residency::Device Storage of one runtime.
     * @see modules/engine/engine/README.md
     */
    class RuntimeScratchSource final : public ScratchBlockSource
    {
        public:
            /**
             * @brief Binds the source to the runtime that allocates.
             * @param runtime Must outlive the source and every block it served.
             */
            explicit RuntimeScratchSource(SushiRuntime::Runtime& runtime) noexcept
                : runtime_(&runtime)
            {

            }

            /** @brief Returns the runtime the source allocates from. */
            [[nodiscard]] SushiRuntime::Runtime& runtime() const noexcept { return *runtime_; }

            /** @brief Returns how many device queues the runtime drives. */
            [[nodiscard]] std::size_t device_count() const override
            {
                return runtime_->advanced().device_count();
            }

            /**
             * @brief Allocates a device Storage of at least @p bytes on the queue.
             * @throws std::runtime_error as Storage's runtime constructor states.
             */
            [[nodiscard]] SushiRuntime::sushi_ptr<Storage> serve(
                std::size_t bytes, std::size_t device_index) override
            {
                return SushiRuntime::make_sushi<Storage>(
                    *runtime_, bytes, SushiRuntime::Residency::Device, device_index);
            }

        private:
            /// @brief The runtime that allocates; never null.
            SushiRuntime::Runtime* runtime_;
    };
} // namespace SushiBLAS
