/**************************************************************************/
/* hand_built_operand.hpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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
 * @file hand_built_operand.hpp
 * @brief Declares HandBuiltOperand, a tensor and the facts of its storage built by hand, so the
 *        record-time operand checks run without a runtime or a queue.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Holds a tensor and the StorageFacts of a storage it does not have.
         *
         * A new operand is dense, row-major and present, and covers its bytes from the start of
         * a made-up allocation; operands in one allocation overlap when their bytes do.
         * @note placed() refers to this object, which must outlive the PlacedTensor.
         */
        class HandBuiltOperand final
        {
            public:
                /**
                 * @brief Builds a dense row-major operand.
                 * @param dims The size of each dimension.
                 * @param dtype The element type.
                 * @param allocation Names the made-up allocation; equal numbers share one.
                 */
                HandBuiltOperand(const std::vector<int64_t>& dims, Core::DataType dtype,
                                 std::size_t allocation)
                    : allocation_(allocation)
                {
                    tensor_.rank = static_cast<int32_t>(dims.size());
                    tensor_.dtype = dtype;
                    tensor_.num_elements = 1;

                    int32_t i = 0;
                    for (const int64_t extent : dims)
                    {
                        tensor_.shape[i++] = extent;
                        tensor_.num_elements *= extent;
                    }

                    derive_strides();
                    refresh_region();
                }

                /** @brief Lays the operand out column-major, with the strides that layout implies. */
                HandBuiltOperand& column_major()
                {
                    tensor_.layout = Core::Layout::ColumnMajor;
                    derive_strides();
                    refresh_region();
                    return *this;
                }

                /** @brief Sets the stride of one dimension, which makes the operand a view. */
                HandBuiltOperand& with_stride(int32_t dimension, int64_t stride)
                {
                    tensor_.strides[dimension] = stride;
                    refresh_region();
                    return *this;
                }

                /** @brief Moves the operand to another made-up allocation. */
                HandBuiltOperand& in_allocation(std::size_t allocation)
                {
                    allocation_ = allocation;
                    refresh_region();
                    return *this;
                }

                /** @brief Starts the operand byte_offset bytes into its allocation. */
                HandBuiltOperand& at_byte_offset(std::size_t byte_offset)
                {
                    byte_offset_ = byte_offset;
                    refresh_region();
                    return *this;
                }

                /** @brief Takes the storage away, as a tensor that was never allocated has none. */
                HandBuiltOperand& without_storage()
                {
                    facts_.present = false;
                    refresh_region();
                    return *this;
                }

                /** @brief Puts the operand on a device without the fp64 aspect. */
                HandBuiltOperand& on_device_without_fp64()
                {
                    facts_.supports_fp64 = false;
                    return *this;
                }

                /** @brief Puts the operand on a device without the fp16 aspect. */
                HandBuiltOperand& on_device_without_fp16()
                {
                    facts_.supports_fp16 = false;
                    return *this;
                }

                /** @brief Returns the number of the made-up allocation the bytes are in. */
                std::size_t allocation() const noexcept
                {
                    return allocation_;
                }

                /** @brief Returns the tensor the checks read the shape from. */
                const Tensor& tensor() const noexcept
                {
                    return tensor_;
                }

                /** @brief Returns the tensor paired with its storage facts. */
                Dispatch::PlacedTensor placed() const
                {
                    return Dispatch::PlacedTensor{tensor_, facts_};
                }

            private:
                /// @brief The spacing of the made-up allocations, wide enough that none touch.
                static constexpr std::size_t K_ALLOCATION_SPACING = std::size_t{1} << 20;

                /** @brief Sets the strides the shape and the layout imply. */
                void derive_strides()
                {
                    tensor_.strides = Tensor::layout_strides(
                        SushiRuntime::span<const int64_t>(tensor_.shape.data(),
                                                          static_cast<std::size_t>(tensor_.rank)),
                        tensor_.layout);
                }

                /** @brief Returns how many elements the operand spans, strides included. */
                int64_t element_span() const
                {
                    if (tensor_.num_elements == 0)
                        return 0;

                    int64_t span = 1;
                    for (int32_t i = 0; i < tensor_.rank; ++i)
                        span += (tensor_.shape[i] - 1) * tensor_.strides[i];
                    return span;
                }

                /** @brief Recomputes the bytes the operand covers from its shape and placement. */
                void refresh_region()
                {
                    if (!facts_.present)
                    {
                        facts_.region = SushiRuntime::BufferRegion{};
                        return;
                    }

                    void* base = reinterpret_cast<void*>(allocation_ * K_ALLOCATION_SPACING);
                    const std::size_t bytes = static_cast<std::size_t>(element_span()) *
                                              Core::bytes_per_element(tensor_.dtype);
                    facts_.region = SushiRuntime::BufferRegion(base, byte_offset_, bytes);
                }

                Tensor tensor_;                     /**< The shape, strides, dtype and layout. */
                Dispatch::StorageFacts facts_{true}; /**< What the checks read of the storage. */
                std::size_t allocation_;            /**< Which made-up allocation the bytes are in. */
                std::size_t byte_offset_ = 0;       /**< Where the bytes start in the allocation. */
        };
    } // namespace Test
} // namespace SushiBLAS
