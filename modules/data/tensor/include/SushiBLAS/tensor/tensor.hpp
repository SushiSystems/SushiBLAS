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
#include <utility>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include <SushiRuntime/memory/resource_region.hpp>

namespace SushiBLAS
{
    namespace Detail
    {
        /**
         * @brief Returns a * b, throwing if the product of two non-negative values overflows int64_t.
         * @pre a >= 0 and b >= 0.
         * @throws std::runtime_error naming what on overflow.
         */
        [[nodiscard]] inline int64_t checked_multiply(int64_t a, int64_t b, const char* what)
        {
            SB_THROW_IF(a != 0 && b > INT64_MAX / a, "{} overflows int64 ({} * {})", what, a, b);
            return a * b;
        }

        /**
         * @brief Returns a + b, throwing if the sum of two non-negative values overflows int64_t.
         * @pre a >= 0 and b >= 0.
         * @throws std::runtime_error naming what on overflow.
         */
        [[nodiscard]] inline int64_t checked_add(int64_t a, int64_t b, const char* what)
        {
            SB_THROW_IF(b > INT64_MAX - a, "{} overflows int64 ({} + {})", what, a, b);
            return a + b;
        }
    } // namespace Detail

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
        std::array<int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK> shape{};
        
        /** @brief The memory step size for each dimension to calculate indices. */
        std::array<int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK> strides{};
        
        /** @brief The number of dimensions in the tensor (e.g., 2 for a matrix). */
        int32_t rank = 0;
        
        /** @brief The type of data stored in the tensor (e.g., FLOAT32). */
        SushiBLAS::Core::DataType dtype = SushiBLAS::Core::DataType::Float32;
        
        /** @brief How the data is arranged in memory (RowMajor or ColumnMajor). */
        SushiBLAS::Core::Layout layout = SushiBLAS::Core::Layout::RowMajor;
        
        /** @brief The total number of elements in the tensor. */
        int64_t num_elements = 0;

        /** @brief A pointer to the memory storage where the actual data is kept. */
        SushiRuntime::sushi_ptr<Storage> storage;

        /**
         * @brief Holds where the first element sits inside the storage, in elements.
         * @see modules/README.md
         */
        int64_t storage_offset = 0;

        /** @brief Default constructor for an empty tensor. */
        Tensor() = default;

        /**
         * @brief Returns the strides a shape implies under a layout.
         * @return The stride of each dimension, in elements; entries past `dims.size()` are 0.
         * @throws std::runtime_error on a rank too high, a negative dimension or an overflowing stride.
         * @see modules/README.md
         */
        static std::array<int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK>
        layout_strides(SushiRuntime::span<const int64_t> dims, Core::Layout l)
        {
            std::array<int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK> out{};

            SB_THROW_IF(dims.size() > SushiBLAS::Core::K_MAX_TENSOR_RANK, "Rank {} exceeds limit of {}", dims.size(), SushiBLAS::Core::K_MAX_TENSOR_RANK);
            for (size_t i = 0; i < dims.size(); ++i)
                SB_THROW_IF(dims[i] < 0, "Dimension {} is negative ({})", i, dims[i]);

            const int r = static_cast<int>(dims.size());
            // The matrix is the trailing two axes, or the whole tensor below rank 2; see modules/README.md.
            const int first_matrix_dim = r - std::min(r, 2);

            int64_t running = 1;
            if (l == Core::Layout::ColumnMajor)
            {
                for (int i = first_matrix_dim; i < r; ++i)
                {
                    out[i] = running;
                    running = Detail::checked_multiply(running, dims[i], "Stride");
                }
            }
            else // RowMajor
            {
                for (int i = r - 1; i >= first_matrix_dim; --i)
                {
                    out[i] = running;
                    running = Detail::checked_multiply(running, dims[i], "Stride");
                }
            }

            // Batch axes, outermost and slowest-varying in either layout.
            for (int i = first_matrix_dim - 1; i >= 0; --i)
            {
                out[i] = running;
                running = Detail::checked_multiply(running, dims[i], "Stride");
            }

            return out;
        }

        /**
         * @brief Creates a Tensor whose strides derive from its dimensions and layout.
         * @param offset The starting position in the storage, in elements.
         * @param l Orders the memory within one matrix; batch axes stay outermost.
         * @throws std::runtime_error on a rank too high, a negative dimension, an overflowing size, or a storage too small.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, SushiRuntime::span<const int64_t> dims,
               Core::DataType d = Core::DataType::Float32,
               int64_t offset = 0, Core::Layout l = Core::Layout::RowMajor)
            : Tensor(std::move(s), dims, layout_strides(dims, l), d, offset, l) {}

        /**
         * @brief Creates a Tensor view whose strides are given instead of derived from its shape.
         * @param strides_in The stride of each dimension, in elements; none within the rank may be negative.
         * @param offset The starting position in the storage, in elements.
         * @throws std::runtime_error on a rank too high, a negative dimension or stride, an overflowing size, or elements past the end of s.
         * @see modules/README.md
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, SushiRuntime::span<const int64_t> dims,
               const std::array<int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK>& strides_in,
               Core::DataType d, int64_t offset, Core::Layout l)
            : dtype(d), layout(l), storage(std::move(s)), storage_offset(offset)
        {
            SB_THROW_IF(dims.size() > SushiBLAS::Core::K_MAX_TENSOR_RANK, "Rank {} exceeds limit of {}", dims.size(), SushiBLAS::Core::K_MAX_TENSOR_RANK);
            rank = static_cast<int32_t>(dims.size());

            int64_t elements = 1;
            for (int i = 0; i < rank; ++i)
            {
                SB_THROW_IF(dims[i] < 0, "Dimension {} is negative ({})", i, dims[i]);
                SB_THROW_IF(strides_in[i] < 0, "Stride of dimension {} is negative ({})", i, strides_in[i]);
                shape[i] = dims[i];
                strides[i] = strides_in[i];
                elements = Detail::checked_multiply(elements, dims[i], "Element count");
            }
            num_elements = elements;

            if (storage)
            {
                SB_THROW_IF(storage_offset < 0, "Storage offset cannot be negative ({})", storage_offset);

                // The element span is checked, because a strided view spans more elements than it holds; see modules/README.md.
                int64_t span = num_elements == 0 ? 0 : 1;
                for (int i = 0; i < rank && num_elements != 0; ++i)
                    span = Detail::checked_add(span, Detail::checked_multiply(shape[i] - 1, strides[i], "Element span"), "Element span");

                const int64_t bpe = static_cast<int64_t>(Core::bytes_per_element(dtype));
                const int64_t end_elements = Detail::checked_add(storage_offset, span, "Storage end");
                size_t required_bytes = static_cast<size_t>(Detail::checked_multiply(end_elements, bpe, "Byte size"));

                SB_THROW_IF(required_bytes > storage->size_bytes(),
                            "Storage capacity exceeded! Required: {} bytes, Available: {} bytes",
                            required_bytes, storage->size_bytes());
            }
        }

        /**
         * @brief Creates a Tensor from an initializer list of dimensions.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::initializer_list<int64_t> dims, 
               Core::DataType d = Core::DataType::Float32,
               int64_t offset = 0, Core::Layout l = Core::Layout::RowMajor)
            : Tensor(std::move(s), SushiRuntime::span<const int64_t>(dims.begin(), dims.end()), d, offset, l) {}

        /**
         * @brief Get the hardware device (GPU/CPU) where this tensor lives.
         * @return The SYCL device object.
         */
        inline sycl::device get_device() const
        {
            SB_THROW_IF(!storage, "Tensor has no valid storage/context");
            return storage->get_context().devices().queue(storage->get_device_index()).get_device();
        }

        /**
         * @brief Returns a pointer to the first element, storage_offset elements past the allocation base.
         * @throws std::runtime_error if the tensor has no storage.
         * @see modules/README.md
         */
        inline void* data()
        {
            return first_element_address();
        }

        /**
         * @brief Returns a read-only pointer to the first element, storage_offset elements past the allocation base.
         * @throws std::runtime_error if the tensor has no storage.
         * @see modules/README.md
         */
        inline const void* data() const
        {
            return first_element_address();
        }

        /**
         * @brief Returns the address of the first element, or nullptr if the tensor has no storage.
         * @see modules/README.md
         */
        inline void* data_or_null()
        {
            return storage ? data() : nullptr;
        }

        /**
         * @brief Returns the read-only address of the first element, or nullptr if the tensor has no storage.
         * @see modules/README.md
         */
        inline const void* data_or_null() const
        {
            return storage ? data() : nullptr;
        }

        /**
         * @brief Get a pointer to the data cast to a specific type.
         * @tparam T The type to cast to (default is float).
         * @return A pointer of type T.
         */
        template<typename T = float>
        inline T* data_as()
        {
            return static_cast<T*>(data());
        }

        /**
         * @brief Get a read-only pointer to the data cast to a specific type.
         * @tparam T The type to cast to (default is float).
         * @return A pointer of type const T.
         */
        template<typename T = float>
        inline const T* data_as() const
        {
            return static_cast<const T*>(data());
        }

        /**
         * @brief Returns how many elements the view spans from its first to its last, inclusive.
         * @return The span in elements; 0 for an empty tensor.
         * @see modules/README.md
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
         * @see modules/README.md
         */
        inline SushiRuntime::Memory::ResourceRegion region() const
        {
            if (!storage) return {};

            const size_t bpe = Core::bytes_per_element(dtype);
            return { storage->allocation_base(),
                     static_cast<size_t>(storage_offset) * bpe,
                     static_cast<size_t>(element_span()) * bpe };
        }

        /**
         * @brief Returns whether the strides match what layout_strides gives for this shape and layout.
         * @see modules/README.md
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
         * @brief Returns whether the first element sits on a K_STORAGE_ALIGNMENT boundary.
         * @return True for every view at storage_offset 0; a view at another offset is aligned
         *         to its element size only.
         * @throws std::runtime_error if the tensor has no storage.
         */
        bool is_aligned() const
        {
            return reinterpret_cast<std::uintptr_t>(data()) % K_STORAGE_ALIGNMENT == 0;
        }

        /**
         * @brief Returns a view with two dimensions swapped; no element moves.
         * @throws std::runtime_error if either index is negative or not below the rank.
         */
        Tensor transpose(int32_t dim0, int32_t dim1) const 
        {
            SB_THROW_IF(dim0 < 0 || dim1 < 0 || dim0 >= rank || dim1 >= rank,
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
            return reshape_to(SushiRuntime::span<const int64_t>(new_dims.begin(), new_dims.end()));
        }

        /**
         * @brief Returns a view of the same elements under dimensions known only at run time.
         * @pre The tensor is contiguous.
         * @throws std::runtime_error on an element count mismatch, a negative dimension or a strided tensor.
         */
        Tensor reshape(const std::vector<int64_t>& new_dims) const
        {
            return reshape_to(SushiRuntime::span<const int64_t>(new_dims.data(), new_dims.size()));
        }

        /**
         * @brief Returns a view that narrows one axis and keeps the parent's strides.
         * @param start Inclusive; negative counts back from the end.
         * @param end Exclusive; negative counts back from the end.
         * @throws std::runtime_error if dim is not a dimension of this tensor.
         * @see modules/README.md
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
            
            std::array<int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK> new_shape_arr = shape;
            new_shape_arr[dim] = end - start;

            SB_LOG_DEBUG("Tensor Sliced: dim = {}, range = [{}, {}), new_offset = {}", 
                        dim, start, end, new_offset);

            return Tensor(this->storage, SushiRuntime::span<const int64_t>(new_shape_arr.data(), static_cast<size_t>(rank)),
                          this->strides, this->dtype, new_offset, this->layout);
        }

    private:
        /**
         * @brief Returns the address storage_offset elements past the allocation base.
         * @throws std::runtime_error if the tensor has no storage or no data pointer.
         */
        inline void* first_element_address() const
        {
            SB_THROW_IF(storage == nullptr, "Accessing data of a tensor with no storage");
            SB_THROW_IF(storage->allocation_base() == nullptr, "Accessing data of a tensor with no data pointer");

            return static_cast<char*>(storage->allocation_base()) + (storage_offset * static_cast<int64_t>(Core::bytes_per_element(dtype)));
        }

        /**
         * @brief Returns a view of the same elements under new_dims.
         * @throws std::runtime_error on an element count mismatch, a negative dimension or a
         *         strided tensor.
         */
        Tensor reshape_to(SushiRuntime::span<const int64_t> new_dims) const
        {
            int64_t new_len = 1;
            for (size_t i = 0; i < new_dims.size(); ++i)
            {
                SB_THROW_IF(new_dims[i] < 0, "Dimension {} is negative ({})", i, new_dims[i]);
                new_len = Detail::checked_multiply(new_len, new_dims[i], "Element count");
            }

            SB_THROW_IF(new_len != this->num_elements,
                        "Element count mismatch. Current: {}, New: {}",
                        this->num_elements, new_len);

            SB_THROW_IF(!is_contiguous(), "Tensor must be contiguous for this operation");

            SB_LOG_DEBUG("Tensor Reshaped: elements={}", new_len);

            return Tensor(this->storage, new_dims, this->dtype, this->storage_offset, this->layout);
        }
    };
} // namespace SushiBLAS
