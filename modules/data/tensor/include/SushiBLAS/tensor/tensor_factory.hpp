/**************************************************************************/
/* tensor_factory.hpp                                                     */
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
 * @file tensor_factory.hpp
 * @brief Declares TensorFactory, which allocates dense tensors from one runtime or from host memory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/block_maker.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @brief Allocates dense tensors of one default layout through a BlockMaker it owns: one
     *        runtime's buffers, or host memory.
     * @see modules/README.md
     */
    class TensorFactory final
    {
        public:
            /** @brief Selects the constructor of a factory that needs no runtime. */
            struct Host
            {

            };

            /**
             * @brief Binds the factory to the runtime that allocates.
             * @param runtime Performs every allocation; it must outlive the factory.
             * @param layout Memory layout of every tensor the factory creates.
             * @throws std::bad_alloc if the block maker cannot be allocated.
             */
            explicit TensorFactory(SushiRuntime::Runtime& runtime,
                                   Core::Layout layout = Core::Layout::RowMajor)
                : maker_(std::make_unique<RuntimeBlockMaker>(runtime)), layout_(layout)
            {

            }

            /**
             * @brief Makes a factory whose storages are host memory no runtime owns.
             * @param layout Memory layout of every tensor the factory creates.
             * @throws std::bad_alloc if the block maker cannot be allocated.
             */
            explicit TensorFactory(Host, Core::Layout layout = Core::Layout::RowMajor)
                : maker_(std::make_unique<HostBlockMaker>()), layout_(layout)
            {

            }

            TensorFactory(const TensorFactory&) = delete;
            TensorFactory& operator=(const TensorFactory&) = delete;
            TensorFactory(TensorFactory&&) = delete;
            TensorFactory& operator=(TensorFactory&&) = delete;
            ~TensorFactory() = default;

            /** @brief Returns the memory layout of every tensor the factory creates. */
            [[nodiscard]] Core::Layout layout() const noexcept { return layout_; }

            /**
             * @brief Returns the byte count a dense tensor of dims and dtype needs; 0 if a dimension is 0.
             * @throws std::runtime_error on a rank too high, a negative dimension or an overflowing count.
             */
            [[nodiscard]] static std::size_t allocation_bytes(SushiRuntime::span<const int64_t> dims,
                                                              Core::DataType dtype);

            /**
             * @brief Creates a dense tensor; a shape with a zero dimension gets no storage.
             * @param residency Where the memory lives; a host factory ignores it.
             * @param device_index Device queue that owns the allocation; 0 on a host factory.
             * @throws std::runtime_error on a rank too high, a negative dimension, an overflowing
             *         byte count, a device index out of range or a failed allocation.
             */
            [[nodiscard]] Tensor create(
                SushiRuntime::span<const int64_t> dims,
                Core::DataType dtype = Core::DataType::Float32,
                SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                std::size_t device_index = 0) const
            {
                const std::size_t bytes = allocation_bytes(dims, dtype);
                const std::size_t device_count = maker_->device_count();
                SB_THROW_IF(device_index >= device_count,
                            "Device index {} is out of range; the runtime has {} device queues",
                            device_index, device_count);

                if (bytes == 0)
                    return Tensor(SushiRuntime::sushi_ptr<Storage>(), dims, dtype, 0, layout_);

                SushiRuntime::sushi_ptr<Storage> storage =
                    SushiRuntime::make_sushi<Storage>(maker_->make(bytes, residency, device_index));
                return Tensor(std::move(storage), dims, dtype, 0, layout_);
            }

        private:
            /// @brief Holds the maker of the block of every created storage.
            std::unique_ptr<BlockMaker> maker_;

            /// @brief Holds the memory layout of every created tensor.
            Core::Layout layout_;
    };
} // namespace SushiBLAS
