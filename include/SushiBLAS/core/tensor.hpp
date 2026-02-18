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
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/storage.hpp>

namespace SushiBLAS 
{
    struct Tensor 
    {
        // Metadata
        std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK> shape{};
        std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK> strides{};
        int32_t rank = 0;
        
        // Logical Size
        int64_t num_elements = 0;

        // Data Ownership
        SushiRuntime::sushi_ptr<Storage> storage;
        int64_t storage_offset = 0;

        Tensor() = default;

        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::span<const int64_t> dims, int64_t offset = 0) 
            : rank(static_cast<int32_t>(dims.size())), storage(s), storage_offset(offset)
        {
            SB_THROW_IF(rank > SushiBLAS::Core::MAX_TENSOR_RANK, "Rank {} exceeds limit of {}", rank, SushiBLAS::Core::MAX_TENSOR_RANK);

            int64_t elements = 1;
            for (int i = rank - 1; i >= 0; --i) 
            {
                shape[i] = dims[i];
                strides[i] = elements;
                elements *= dims[i];
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

        // Public API
        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::initializer_list<int64_t> dims, int64_t offset = 0)
            : Tensor(s, std::span<const int64_t>(dims.begin(), dims.end()), offset) {}

        /**
         * @brief Returns the SYCL device where this tensor is allocated.
         */
        inline sycl::device get_device() const 
        {
            SB_THROW_IF(!storage || !storage->allocator, "Tensor has no valid storage/allocator");
            return storage->allocator->get_device_of(storage->data_ptr);
        }

        // Data Access
        inline float* data() const 
        {
            SB_THROW_IF(storage == nullptr, "Accessing data of a tensor with no storage");
            SB_THROW_IF(storage->data_ptr == nullptr, "Accessing data of a tensor with no data pointer");

            return static_cast<float*>(storage->data_ptr) + storage_offset;
        }

        // Contiguity Check
        bool is_contiguous() const 
        {
            if (num_elements == 0) return true;
            int64_t s = 1;
            for (int i = rank - 1; i >= 0; --i) 
            {
                if (shape[i] != 1) 
                {
                    if (strides[i] != s) return false;
                    s *= shape[i];
                }
            }
            return true;
        }

        // Alignment Check
        bool is_aligned() const 
        {
            SB_THROW_IF(data() == nullptr, "Alignment check on a tensor with no data pointer (Storage might be null)");
            return (reinterpret_cast<uintptr_t>(data()) % SushiRuntime::Core::DEFAULT_ALIGNMENT == 0);
        }

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

            return Tensor(this->storage, dims_span, this->storage_offset);
        }

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

            return Tensor(this->storage, std::span(new_shape_arr.data(), rank), new_offset);
        }
    };
} // namespace SushiBLAS