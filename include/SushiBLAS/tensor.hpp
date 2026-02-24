/**************************************************************************/
/* tensor.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include <span>
#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/storage.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>

namespace SushiBLAS 
{
    /**
     * @struct Tensor
     * @brief A structure that represents a multi-dimensional array of data.
     * 
     * The Tensor structure stores metadata like shape and strides. 
     * It does not own the memory directly but uses a Storage object to access data.
     */
    struct Tensor 
    {
        /** @brief The size of each dimension (e.g., {rows, columns}). */
        std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK> shape{};
        
        /** @brief The memory step size for each dimension to calculate indices. */
        std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK> strides{};
        
        /** @brief The number of dimensions in the tensor (e.g., 2 for a matrix). */
        int32_t rank = 0;
        
        /** @brief The type of data stored in the tensor (e.g., FLOAT32). */
        SushiBLAS::Core::DataType dtype = SushiBLAS::Core::DataType::FLOAT32;
        
        /** @brief How the data is arranged in memory (ROW_MAJOR or COLUMN_MAJOR). */
        SushiBLAS::Core::Layout layout = SushiBLAS::Core::Layout::ROW_MAJOR;
        
        /** @brief The total number of elements in the tensor. */
        int64_t num_elements = 0;

        /** @brief A pointer to the memory storage where the actual data is kept. */
        SushiRuntime::sushi_ptr<Storage> storage;
        
        /** @brief The starting index in the storage where this tensor's data begins. */
        int64_t storage_offset = 0;

        /** @brief Default constructor for an empty tensor. */
        Tensor() = default;

        /**
         * @brief Create a new Tensor with specific dimensions.
         * 
         * @param s The storage object to use for data.
         * @param dims A list of sizes for each dimension.
         * @param offset The starting position in the storage.
         * @param l The memory layout (default is ROW_MAJOR).
         * @throws std::runtime_error If the rank is too high or storage is too small.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::span<const int64_t> dims, 
               int64_t offset = 0, Core::Layout l = Core::Layout::ROW_MAJOR) 
            : rank(static_cast<int32_t>(dims.size())), layout(l), storage(s), storage_offset(offset)
        {
            SB_THROW_IF(static_cast<size_t>(rank) > SushiBLAS::Core::MAX_TENSOR_RANK, "Rank {} exceeds limit of {}", rank, SushiBLAS::Core::MAX_TENSOR_RANK);

            int64_t elements = 1;
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                for (int i = rank - 1; i >= 0; --i) 
                {
                    shape[i] = dims[i];
                    strides[i] = elements;
                    elements *= dims[i];
                }
            } 
            else // COLUMN_MAJOR
            {
                for (int i = 0; i < rank; ++i) 
                {
                    shape[i] = dims[i];
                    strides[i] = elements;
                    elements *= dims[i];
                }
            }
            num_elements = elements;

            if (storage) 
            {
                SB_THROW_IF(storage_offset < 0, "Storage offset cannot be negative ({})", storage_offset);

                size_t required_bytes = (static_cast<size_t>(num_elements) + storage_offset) * sizeof(float);
                
                SB_THROW_IF(required_bytes > storage->size_bytes, 
                            "Storage capacity exceeded! Required: {} bytes, Available: {} bytes", 
                            required_bytes, storage->size_bytes);
            }
        }

        /**
         * @brief Create a new Tensor using an initializer list for dimensions.
         * 
         * @param s The storage object to use.
         * @param dims Example: {3, 3} for a 3x3 matrix.
         * @param offset Starting position in storage.
         * @param l Memory layout strategy.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::initializer_list<int64_t> dims, 
               int64_t offset = 0, Core::Layout l = Core::Layout::ROW_MAJOR)
            : Tensor(s, std::span<const int64_t>(dims.begin(), dims.end()), offset, l) {}

        /**
         * @brief Get the hardware device (GPU/CPU) where this tensor lives.
         * @return The SYCL device object.
         */
        inline sycl::device get_device() const 
        {
            SB_THROW_IF(!storage || !storage->allocator, "Tensor has no valid storage/allocator");
            return storage->allocator->get_device_of(storage->data_ptr);
        }

        /**
         * @brief Get a raw pointer to the tensor's data.
         * 
         * This calculates the address using the storage pointer and the offset.
         * @return A void pointer to the start of the data.
         */
        inline void* data() const 
        {
            SB_THROW_IF(storage == nullptr, "Accessing data of a tensor with no storage");
            SB_THROW_IF(storage->data_ptr == nullptr, "Accessing data of a tensor with no data pointer");

            // Offset is in terms of elements, so we must scale by byte size of the dtype
            size_t element_size = (dtype == Core::DataType::FLOAT64) ? 8 : 4; 
            return static_cast<char*>(storage->data_ptr) + (storage_offset * element_size);
        }

        /**
         * @brief Get a pointer to the data cast to a specific type.
         * @tparam T The type to cast to (default is float).
         * @return A pointer of type T.
         */
        template<typename T = float>
        inline T* data_as() const 
        {
            return static_cast<T*>(data());
        }

        /**
         * @brief Check if the tensor data is stored continuously in memory.
         * @return True if it is contiguous, false otherwise.
         */
        bool is_contiguous() const 
        {
            if (num_elements == 0) return true;
            
            int64_t expected_stride = 1;
            if (layout == Core::Layout::ROW_MAJOR) 
            {
                for (int i = rank - 1; i >= 0; --i) 
                {
                    if (shape[i] != 1) 
                    {
                        if (strides[i] != expected_stride) return false;
                        expected_stride *= shape[i];
                    }
                }
            }
            else // COLUMN_MAJOR
            {
                for (int i = 0; i < rank; ++i) 
                {
                    if (shape[i] != 1) 
                    {
                        if (strides[i] != expected_stride) return false;
                        expected_stride *= shape[i];
                    }
                }
            }
            return true;
        }


        /**
         * @brief Check if the data pointer is correctly aligned for high performance.
         * @return True if aligned, false otherwise.
         */
        bool is_aligned() const 
        {
            SB_THROW_IF(data() == nullptr, "Alignment check on a tensor with no data pointer (Storage might be null)");
            return (reinterpret_cast<uintptr_t>(data()) % SushiRuntime::Core::DEFAULT_ALIGNMENT == 0);
        }

        /**
         * @brief Swap two dimensions of the tensor (e.g., for matrix transpose).
         * 
         * This is a "view" operation; it does not move data in memory.
         * @param dim0 The first dimension index.
         * @param dim1 The second dimension index.
         * @return A new Tensor view with swapped dimensions.
         */
        Tensor transpose(int32_t dim0, int32_t dim1) const 
        {
            SB_THROW_IF(dim0 >= rank || dim1 >= rank, 
                        "Invalid dimensions ({}, {}) for rank {}", dim0, dim1, rank);

            Tensor t = *this;
            std::swap(t.shape[dim0], t.shape[dim1]);
            std::swap(t.strides[dim0], t.strides[dim1]);

            SB_LOG_DEBUG("Tensor Transposed: dim {} <-> dim {}", dim0, dim1);

            return t;
        }

        /**
         * @brief Change the shape of the tensor without changing its data.
         * 
         * The tensor must be contiguous for this to work correctly.
         * @param new_dims The new dimensions for the tensor.
         * @return A new Tensor view with the new shape.
         */
        Tensor reshape(std::initializer_list<int64_t> new_dims) const 
        {
            std::span<const int64_t> dims_span(new_dims.begin(), new_dims.end());
            
            int64_t new_len = 1;
            for(auto d : dims_span) new_len *= d;

            SB_THROW_IF(new_len != this->num_elements, 
                        "Element count mismatch. Current: {}, New: {}", 
                        this->num_elements, new_len);

            SB_THROW_IF(!is_contiguous(), "Tensor must be contiguous for this operation");

            SB_LOG_DEBUG("Tensor Reshaped: elements={}", new_len);

            return Tensor(this->storage, dims_span, this->storage_offset, this->layout);
        }

        /**
         * @brief Take a "slice" or sub-section of the tensor.
         * 
         * @param dim The dimension to slice (e.g., slice certain rows).
         * @param start The starting index (inclusive).
         * @param end The ending index (exclusive).
         * @return A new Tensor view representing the slice.
         */
        Tensor slice(int32_t dim, int64_t start, int64_t end) const 
        {
            SB_THROW_IF(dim < 0 || dim >= rank, "Invalid dimension {}", dim);
            
            int64_t dim_size = shape[dim];
            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size;

            start = std::max<int64_t>(0, std::min(start, dim_size));
            end = std::max<int64_t>(start, std::min(end, dim_size));

            int64_t new_offset = this->storage_offset + (start * strides[dim]);
            
            std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK> new_shape_arr = shape;
            new_shape_arr[dim] = end - start;

            SB_LOG_DEBUG("Tensor Sliced: dim = {}, range = [{}, {}), new_offset = {}", 
                        dim, start, end, new_offset);

            return Tensor(this->storage, std::span(new_shape_arr.data(), rank), new_offset, this->layout);
        }
    };
} // namespace SushiBLAS