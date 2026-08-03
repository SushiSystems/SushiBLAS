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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
         * @brief Where this tensor's first element sits inside its storage, **in elements**.
         *
         * Elements, never bytes. `data()` scales it by `bytes_per_element(dtype)`, so a view
         * is by construction at a whole number of elements from a 64-byte-aligned allocation
         * base and therefore always naturally aligned for its own element type - an fp32 view
         * cannot land on an odd byte. Anything that computes an offset (`slice()`, a memory
         * planner packing an arena) must count in elements too, and a planner packing mixed
         * dtypes into one allocation has to round each tensor's start up to a multiple of its
         * own element width itself; this class cannot do it for the planner, because by the
         * time a Tensor exists the offset has already been chosen.
         */
        int64_t storage_offset = 0;

        /** @brief Default constructor for an empty tensor. */
        Tensor() = default;

        /**
         * @brief The strides a shape implies under a layout, as this class defines it.
         *
         * The single definition of "contiguous" for the whole library: the constructor lays
         * tensors out with it and `is_contiguous()` checks against it, so the two cannot
         * drift apart.
         *
         * **`layout` describes the memory order inside one matrix, not the whole tensor.**
         * For rank <= 2 that is the entire tensor and this is the familiar rule: row-major
         * strides grow from the last dimension inward, column-major from the first. For rank
         * >= 3 the trailing two axes are the matrix and take the layout's order, while every
         * leading (batch) axis is outermost and slowest-varying **whatever the layout is** -
         * so a rank-3 `{b, m, n}` tensor has batch stride `m * n` in both layouts.
         *
         * That asymmetry is deliberate. It is the only reading consistent with how the batch
         * stride is consumed (`A.strides[rank - 3]` is handed to the GEMM/SYRK/TRSM drivers as
         * the distance between consecutive batch slices) and with every batched BLAS API,
         * which takes a `stride_a` alongside a per-slice `lda` rather than folding the batch
         * into the layout. Treating the batch axis as just another column-major dimension -
         * which is what the old `for (i = 0; i < rank; ++i)` loop did - gave a rank-3
         * column-major tensor `strides[0] == 1`, so the driver read slice 0 for every batch
         * element and a batched column-major GEMM silently computed the same product `b`
         * times.
         *
         * @param dims The extent of each dimension.
         * @param l The memory layout to lay the dimensions out under.
         * @return The stride of each dimension, in elements; entries past `dims.size()` are 0.
         */
        static std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK>
        layout_strides(std::span<const int64_t> dims, Core::Layout l)
        {
            std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK> out{};

            const int r = static_cast<int>(dims.size());
            // The matrix inside a batch element: the trailing two axes, or the whole
            // tensor when there are fewer than two of them.
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
         * @brief Create a new Tensor with specific dimensions.
         *
         * @param s The storage object to use for data.
         * @param dims A list of sizes for each dimension.
         * @param d The data type (default is FLOAT32).
         * @param offset The starting position in the storage, **in elements** (see
         *        @ref storage_offset).
         * @param l The memory layout (default is ROW_MAJOR). It orders the memory *within*
         *        one matrix only; for rank >= 3 the batch axes are outermost regardless -
         *        see @ref layout_strides for the full rule and why.
         * @throws std::runtime_error If the rank is too high or storage is too small.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::span<const int64_t> dims,
               Core::DataType d = Core::DataType::FLOAT32,
               int64_t offset = 0, Core::Layout l = Core::Layout::ROW_MAJOR)
            : rank(static_cast<int32_t>(dims.size())), dtype(d), layout(l), storage(s), storage_offset(offset)
        {
            SB_THROW_IF(static_cast<size_t>(rank) > SushiBLAS::Core::MAX_TENSOR_RANK, "Rank {} exceeds limit of {}", rank, SushiBLAS::Core::MAX_TENSOR_RANK);

            strides = layout_strides(dims, layout);

            int64_t elements = 1;
            for (int i = 0; i < rank; ++i)
            {
                shape[i] = dims[i];
                elements *= dims[i];
            }
            num_elements = elements;

            if (storage)
            {
                SB_THROW_IF(storage_offset < 0, "Storage offset cannot be negative ({})", storage_offset);

                const size_t bpe = Core::bytes_per_element(dtype);
                size_t required_bytes = (static_cast<size_t>(num_elements) + storage_offset) * bpe;

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
         * @param d The data type.
         * @param offset Starting position in storage.
         * @param l Memory layout strategy.
         */
        Tensor(SushiRuntime::sushi_ptr<Storage> s, std::initializer_list<int64_t> dims, 
               Core::DataType d = Core::DataType::FLOAT32,
               int64_t offset = 0, Core::Layout l = Core::Layout::ROW_MAJOR)
            : Tensor(s, std::span<const int64_t>(dims.begin(), dims.end()), d, offset, l) {}

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
         * @brief Get a raw pointer to the tensor's data.
         *
         * This calculates the address using the storage pointer and the offset.
         *
         * **This, not `storage->allocation_base()`, is the pointer an operation hands to a
         * kernel.** The two differ by `storage_offset` elements, which is zero for a
         * freshly allocated tensor and non-zero for every view - a slice, a reshape, or a
         * tensor a memory planner placed inside a shared arena.
         *
         * @return A void pointer to the start of the data.
         * @throws std::runtime_error If the tensor has no storage.
         */
        inline void* data() const
        {
            SB_THROW_IF(storage == nullptr, "Accessing data of a tensor with no storage");
            SB_THROW_IF(storage->allocation_base() == nullptr, "Accessing data of a tensor with no data pointer");

            // Offset is in terms of elements, so we must scale by byte size of the dtype
            return static_cast<char*>(storage->allocation_base()) + (storage_offset * static_cast<int64_t>(Core::bytes_per_element(dtype)));
        }

        /**
         * @brief The null-tolerant form of @ref data().
         *
         * The operation drivers treat a tensor with no storage as "nothing to address"
         * rather than as an error, and every one of them used to spell that as
         * `t.storage ? t.storage->data_ptr : nullptr` - the exact expression that dropped
         * `storage_offset`. This is that idiom, offset-correct, in one place.
         *
         * @return The address of the first element, or nullptr if the tensor has no storage.
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
         * @brief The byte interval this view occupies, for SushiRuntime's dependency tracker.
         *
         * Keyed on the allocation base (the tracker's identity for "same buffer") and
         * narrowed to `[storage_offset, storage_offset + num_elements)` scaled to bytes, so
         * two disjoint views of one allocation - what a memory planner produces when it packs
         * an arena - are correctly seen as *not* conflicting, while two overlapping views
         * still are. Passing the bare base pointer instead, as every driver used to, made
         * every tensor in an arena alias every other one.
         *
         * @return The region, or an empty (base-less) region if the tensor has no storage.
         *
         * @warning The interval is derived from `num_elements`, so it is only a true bound on
         *          what a kernel touches for a tensor whose strides match its shape - which is
         *          every tensor this library's operations accept today.
         */
        inline SushiRuntime::Core::ResourceRegion region() const
        {
            if (!storage) return {};

            const size_t bpe = Core::bytes_per_element(dtype);
            return { storage->allocation_base(),
                     static_cast<size_t>(storage_offset) * bpe,
                     static_cast<size_t>(num_elements) * bpe };
        }

        /**
         * @brief Check if the tensor data is stored continuously in memory.
         *
         * "Continuous" means "laid out exactly as this tensor's layout would lay it out",
         * so the answer is a comparison against @ref layout_strides rather than a second,
         * independently written walk over the dimensions - the two used to be written
         * separately, which is how the rank-3 column-major batch stride could be changed in
         * one and not the other.
         *
         * Axes of extent 1 are exempt: their stride is unobservable, so a tensor is not
         * declared non-contiguous over a stride nothing can ever use.
         *
         * @return True if it is contiguous, false otherwise.
         */
        bool is_contiguous() const
        {
            if (num_elements == 0) return true;

            const auto expected = layout_strides(std::span<const int64_t>(shape.data(), static_cast<size_t>(rank)), layout);
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
            std::span<const int64_t> dims_span(new_dims.begin(), new_dims.end());
            
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

            return Tensor(this->storage, std::span(new_shape_arr.data(), rank), this->dtype, new_offset, this->layout);
        }
    };
} // namespace SushiBLAS
