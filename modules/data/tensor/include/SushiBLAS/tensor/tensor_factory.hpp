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
 * @file tensor_factory.hpp
 * @brief Declares TensorFactory, which allocates dense tensors from one runtime context.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    /**
     * @brief Allocates dense tensors of one default layout from one runtime context.
     * @see modules/README.md
     */
    class TensorFactory final
    {
        public:
            /**
             * @brief Binds the factory to the context that allocates.
             * @param context Performs every allocation; it must outlive the factory and its tensors.
             * @param layout Memory layout of every tensor the factory creates.
             */
            explicit TensorFactory(SushiRuntime::Execution::RuntimeContext& context,
                                   Core::Layout layout = Core::Layout::RowMajor) noexcept
                : context_(&context), layout_(layout)
            {

            }

            TensorFactory(const TensorFactory&) = delete;
            TensorFactory& operator=(const TensorFactory&) = delete;
            TensorFactory(TensorFactory&&) = delete;
            TensorFactory& operator=(TensorFactory&&) = delete;
            ~TensorFactory() = default;

            /** @brief Returns the memory layout of every tensor the factory creates. */
            [[nodiscard]] Core::Layout get_layout() const noexcept { return layout_; }

            /** @brief Returns the runtime context that allocates. */
            [[nodiscard]] SushiRuntime::Execution::RuntimeContext& get_context() const noexcept
            {
                return *context_;
            }

            /**
             * @brief Returns the byte count a dense tensor of dims and dtype needs; 0 if a dimension is 0.
             * @throws std::runtime_error on a rank too high, a negative dimension or an overflowing count.
             */
            [[nodiscard]] static std::size_t allocation_bytes(SushiRuntime::span<const int64_t> dims,
                                                              Core::DataType dtype)
            {
                SB_THROW_IF(dims.size() > Core::K_MAX_TENSOR_RANK, "Rank {} exceeds limit of {}",
                            dims.size(), Core::K_MAX_TENSOR_RANK);

                int64_t elements = 1;
                for (std::size_t axis = 0; axis < dims.size(); ++axis)
                {
                    SB_THROW_IF(dims[axis] < 0, "Dimension {} is negative ({})", axis, dims[axis]);
                    elements = Detail::checked_multiply(elements, dims[axis], "Element count");
                }

                const int64_t bpe = static_cast<int64_t>(Core::bytes_per_element(dtype));
                return static_cast<std::size_t>(Detail::checked_multiply(elements, bpe, "Byte size"));
            }

            /**
             * @brief Creates a dense tensor; a shape with a zero dimension gets no storage.
             * @param device_index Device queue that owns the allocation, below the context's device_count().
             * @throws std::runtime_error on a rank too high, a negative dimension, an overflowing
             *         byte count, a device index out of range or a failed allocation.
             */
            [[nodiscard]] Tensor create(
                SushiRuntime::span<const int64_t> dims,
                Core::DataType dtype = Core::DataType::Float32,
                SushiRuntime::API::Residency residency = SushiRuntime::API::Residency::Shared,
                std::size_t device_index = 0) const
            {
                const std::size_t bytes = allocation_bytes(dims, dtype);
                SB_THROW_IF(device_index >= context_->devices().count(),
                            "Device index {} is out of range; the runtime context has {} device queues",
                            device_index, context_->devices().count());

                if (bytes == 0)
                    return Tensor(SushiRuntime::sushi_ptr<Storage>(), dims, dtype, 0, layout_);

                SushiRuntime::sushi_ptr<Storage> storage =
                    SushiRuntime::make_sushi<Storage>(*context_, bytes, residency, device_index);
                return Tensor(std::move(storage), dims, dtype, 0, layout_);
            }

        private:
            /// @brief The runtime context that allocates; never null.
            SushiRuntime::Execution::RuntimeContext* context_;

            /// @brief The memory layout of every created tensor.
            Core::Layout layout_;
    };
} // namespace SushiBLAS
