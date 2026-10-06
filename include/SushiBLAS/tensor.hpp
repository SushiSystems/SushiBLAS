/**************************************************************************/
/* tensor.hpp                                                             */
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
 * @file tensor.hpp
 * @brief Declares Tensor, a typed, strided view over a Storage.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/storage.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiRuntime/core/resource_region.hpp>

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

        /**
         * @brief Holds where the first element sits inside the storage, in elements.
         * @see include/SushiBLAS/README.md
         */
        int64_t storage_offset = 0;

        /** @brief Default constructor for an empty tensor. */
        Tensor() = default;

        /**
         * @brief Returns the strides a shape implies under a layout.
         * @return The stride of each dimension, in elements; entries past `dims.size()` are 0.
         * @see include/SushiBLAS/README.md
         */
        static std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK>
        layout_strides(SushiRuntime::span<const int64_t> dims, Core::Layout l)
        {
            std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK> out{};

            const int r = static_cast<int>(dims.size());
            // The matrix is the trailing two axes, or the whole tensor below rank 2; see include/SushiBLAS/README.md.
            const int first_matrix_dim = r - std::min(r, 2);

            int64_t running = 1;
            if (l == Core::Layout::COLUMN_MAJOR)
            {
                for (int i = first_matrix_dim; i < r; ++i)
                {
                    out[i] = running;
                    running *= dims[i];
                }
            }
            else // ROW_MAJOR
            {
                for (int i = r - 1; i >= first_matrix_dim; --i)
                {
                    out[i] = running;
                    running *= dims[i];
                }
            }

            // Batch axes, outermost and slowest-varying in either layout.
            for (int i = first_matrix_dim - 1; i >= 0; --i)
            {
                out[i] = running;
                running *= dims[i];
            }

            return out;
        }

        /**
         * @brief Creates a Tensor whose strides derive from its dimensions and layout.
         * @param offset The starting position in the storage, in elements.
         * @param l Orders the memory within one matrix; batch axes stay outermost.
         * @throws std::runtime_error if the rank is too high or the storage is too small.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, SushiRuntime::span<const int64_t> dims,
               Core::DataType d = Core::DataType::FLOAT32,
               int64_t offset = 0, Core::Layout l = Core::Layout::ROW_MAJOR)
            : Tensor(s, dims, layout_strides(dims, l), d, offset, l) {}

        /**
         * @brief Creates a Tensor view whose strides are given instead of derived from its shape.
         * @param strides_in The stride of each dimension, in elements; none within the rank may be negative.
         * @param offset The starting position in the storage, in elements.
         * @throws std::runtime_error on a rank too high, a negative stride, or elements past the end of s.
         * @see include/SushiBLAS/README.md
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, SushiRuntime::span<const int64_t> dims,
               const std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK>& strides_in,
               Core::DataType d, int64_t offset, Core::Layout l)
            : rank(static_cast<int32_t>(dims.size())), dtype(d), layout(l), storage(s), storage_offset(offset)
        {
            SB_THROW_IF(static_cast<size_t>(rank) > SushiBLAS::Core::MAX_TENSOR_RANK, "Rank {} exceeds limit of {}", rank, SushiBLAS::Core::MAX_TENSOR_RANK);

            int64_t elements = 1;
            for (int i = 0; i < rank; ++i)
            {
                SB_THROW_IF(strides_in[i] < 0, "Stride of dimension {} is negative ({})", i, strides_in[i]);
                shape[i] = dims[i];
                strides[i] = strides_in[i];
                elements *= dims[i];
            }
            num_elements = elements;

            if (storage)
            {
                SB_THROW_IF(storage_offset < 0, "Storage offset cannot be negative ({})", storage_offset);

                // element_span() is checked, because a strided view spans more elements than it holds; see include/SushiBLAS/README.md.
                const size_t bpe = Core::bytes_per_element(dtype);
                size_t required_bytes = (static_cast<size_t>(storage_offset) + static_cast<size_t>(element_span())) * bpe;

                SB_THROW_IF(required_bytes > storage->size_bytes,
                            "Storage capacity exceeded! Required: {} bytes, Available: {} bytes",
                            required_bytes, storage->size_bytes);
            }
        }

        /**
         * @brief Creates a Tensor from an initializer list of dimensions.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::initializer_list<int64_t> dims, 
               Core::DataType d = Core::DataType::FLOAT32,
               int64_t offset = 0, Core::Layout l = Core::Layout::ROW_MAJOR)
            : Tensor(s, SushiRuntime::span<const int64_t>(dims.begin(), dims.end()), d, offset, l) {}

        /**
         * @brief Get the hardware device (GPU/CPU) where this tensor lives.
         * @return The SYCL device object.
         */
        inline sycl::device get_device() const
        {
            SB_THROW_IF(!storage || !storage->allocator, "Tensor has no valid storage/allocator");
            return storage->allocator->get_device_of(storage->allocation_base());
        }

        /**
         * @brief Returns a pointer to the first element, storage_offset elements past the allocation base.
         * @throws std::runtime_error if the tensor has no storage.
         * @see include/SushiBLAS/README.md
         */
        inline void* data() const
        {
            SB_THROW_IF(storage == nullptr, "Accessing data of a tensor with no storage");
            SB_THROW_IF(storage->allocation_base() == nullptr, "Accessing data of a tensor with no data pointer");

            // Offset is in terms of elements, so we must scale by byte size of the dtype
            return static_cast<char*>(storage->allocation_base()) + (storage_offset * static_cast<int64_t>(Core::bytes_per_element(dtype)));
        }

        /**
         * @brief Returns the address of the first element, or nullptr if the tensor has no storage.
         * @see include/SushiBLAS/README.md
         */
        inline void* data_or_null() const
        {
            return storage ? data() : nullptr;
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
         * @brief Returns how many elements the view spans from its first to its last, inclusive.
         * @return The span in elements; 0 for an empty tensor.
         * @see include/SushiBLAS/README.md
         */
        [[nodiscard]] inline int64_t element_span() const
        {
            if (num_elements == 0) return 0;

            int64_t span = 1;
            for (int i = 0; i < rank; ++i) span += (shape[i] - 1) * strides[i];
            return span;
        }

        /**
         * @brief Returns the byte interval this view occupies, keyed on the allocation base.
         * @return The region, or an empty base-less region if the tensor has no storage.
         * @see include/SushiBLAS/README.md
         */
        inline SushiRuntime::Core::ResourceRegion region() const
        {
            if (!storage) return {};

            const size_t bpe = Core::bytes_per_element(dtype);
            return { storage->allocation_base(),
                     static_cast<size_t>(storage_offset) * bpe,
                     static_cast<size_t>(element_span()) * bpe };
        }

        /**
         * @brief Returns whether the strides match what layout_strides gives for this shape and layout.
         * @see include/SushiBLAS/README.md
         */
        bool is_contiguous() const
        {
            if (num_elements == 0) return true;

            const auto expected = layout_strides(SushiRuntime::span<const int64_t>(shape.data(), static_cast<size_t>(rank)), layout);
            for (int i = 0; i < rank; ++i)
            {
                if (shape[i] != 1 && strides[i] != expected[i]) return false;
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
            SushiRuntime::span<const int64_t> dims_span(new_dims.begin(), new_dims.end());
            
            int64_t new_len = 1;
            for(auto d : dims_span) new_len *= d;

            SB_THROW_IF(new_len != this->num_elements, 
                        "Element count mismatch. Current: {}, New: {}", 
                        this->num_elements, new_len);

            SB_THROW_IF(!is_contiguous(), "Tensor must be contiguous for this operation");

            SB_LOG_DEBUG("Tensor Reshaped: elements={}", new_len);

            return Tensor(this->storage, dims_span, this->dtype, this->storage_offset, this->layout);
        }

        /**
         * @brief Returns a view that narrows one axis and keeps the parent's strides.
         * @param start Inclusive; negative counts back from the end.
         * @param end Exclusive; negative counts back from the end.
         * @throws std::runtime_error if dim is not a dimension of this tensor.
         * @see include/SushiBLAS/README.md
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

            return Tensor(this->storage, SushiRuntime::span<const int64_t>(new_shape_arr.data(), static_cast<size_t>(rank)),
                          this->strides, this->dtype, new_offset, this->layout);
        }
    };
} // namespace SushiBLAS
