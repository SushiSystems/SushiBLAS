/****************************************************************/
/* tensor.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
         * @param dims The extent of each dimension. `SushiRuntime::span`, not `std::span`:
         *        the two are the same type wherever `std::span` exists (SushiRuntime's
         *        span.hpp aliases it at C++20 and supplies its own subset below), so this
         *        signature is unchanged for a C++20 caller while still compiling on the
         *        C++17 baseline the rest of the stack is built to. Every span in this
         *        header is spelled this way for that reason, and none of them crosses the
         *        compiled ABI - they are all inline.
         * @param l The memory layout to lay the dimensions out under.
         * @return The stride of each dimension, in elements; entries past `dims.size()` are 0.
         */
        static std::array<int64_t, SushiBLAS::Core::MAX_TENSOR_RANK>
        layout_strides(SushiRuntime::span<const int64_t> dims, Core::Layout l)
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
        Tensor(SushiRuntime::sushi_ptr<Storage> s, SushiRuntime::span<const int64_t> dims,
               Core::DataType d = Core::DataType::FLOAT32,
               int64_t offset = 0, Core::Layout l = Core::Layout::ROW_MAJOR)
            : Tensor(s, dims, layout_strides(dims, l), d, offset, l) {}

        /**
         * @brief Create a Tensor view whose strides are given rather than derived from its
         * shape.
         *
         * The construction path for a view that is **not contiguous**: a slice along an
         * inner axis keeps its parent's strides, because that is where its elements actually
         * are. The shape-derived constructor above cannot express that - it lays the
         * dimensions out afresh, which for such a slice describes memory the slice does not
         * occupy - and `slice()` used to go through it, which is exactly the defect this
         * overload exists to make unspellable.
         *
         * Prefer the shape-derived constructor for anything being allocated; this one is for
         * describing memory that already has a layout.
         *
         * @param s The storage object to use for data.
         * @param dims The extent of each dimension.
         * @param strides_in The stride of each dimension, in elements, in the same
         *        fixed-width form @ref layout_strides returns and @ref strides holds -
         *        entries past `dims.size()` are unused, exactly as they are in @ref shape.
         *        A fixed-width array rather than a span so that a stride count and a
         *        dimension count cannot disagree; every entry within the rank must be
         *        non-negative, since a negative one would make @ref element_span, and
         *        therefore @ref region, describe the wrong interval.
         * @param d The data type.
         * @param offset The starting position in the storage, **in elements** (see
         *        @ref storage_offset).
         * @param l The memory layout this view is *interpreted* under. It no longer decides
         *        the strides, but it still tells a BLAS driver how to read the view.
         * @throws std::runtime_error If the rank is too high, a stride is negative, or the
         *         described elements reach past the end of @p s.
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

                // element_span(), not num_elements: a strided view spreads its elements over
                // a wider interval than it has elements, and it is the wider interval that
                // has to fit inside the allocation.
                const size_t bpe = Core::bytes_per_element(dtype);
                size_t required_bytes = (static_cast<size_t>(storage_offset) + static_cast<size_t>(element_span())) * bpe;

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
         * @brief How many elements this view spans from its first to its last, inclusive.
         *
         * Not `num_elements`. For a contiguous tensor the two agree, which is why the
         * distinction went unnoticed - but a strided view (a slice along an inner axis, or a
         * transpose of one) holds `num_elements` elements scattered across a *wider*
         * interval, and it is the wider interval that says which bytes the view can reach.
         * `1 + sum((shape[i] - 1) * strides[i])` is the distance from the first element to
         * the last, which bounds that interval exactly.
         *
         * Used by the capacity check and by @ref region, both of which are questions about
         * reachable memory rather than about element count.
         *
         * @return The span in elements; 0 for an empty tensor.
         */
        [[nodiscard]] inline int64_t element_span() const
        {
            if (num_elements == 0) return 0;

            int64_t span = 1;
            for (int i = 0; i < rank; ++i) span += (shape[i] - 1) * strides[i];
            return span;
        }

        /**
         * @brief The byte interval this view occupies, for SushiRuntime's dependency tracker.
         *
         * Keyed on the allocation base (the tracker's identity for "same buffer") and
         * narrowed to the view's own byte interval, so two disjoint views of one allocation -
         * what a memory planner produces when it packs an arena - are correctly seen as *not*
         * conflicting, while two overlapping views still are. Passing the bare base pointer
         * instead, as every driver used to, made every tensor in an arena alias every other
         * one.
         *
         * The length is @ref element_span rather than `num_elements`, so the bound stays true
         * for a strided view as well as a contiguous one. Stopping at the element count would
         * under-declare what the view reaches, and the tracker would then emit no edge
         * against whatever sits in the gaps between its elements.
         *
         * @return The region, or an empty (base-less) region if the tensor has no storage.
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
         * @brief Take a "slice" or sub-section of the tensor.
         *
         * A view: it moves the offset and shortens one axis, and **keeps the parent's
         * strides**, because narrowing an axis does not move any of the elements that remain.
         *
         * That last part is the whole content of this function and it used to be wrong. The
         * slice was built through the shape-derived constructor, which recomputes contiguous
         * strides from the new shape - so for `dim == 0` the answer happened to come out
         * right (shortening the outermost axis of a contiguous tensor leaves every inner
         * stride unchanged) and for every other `dim` it came out wrong. A 4x4 row-major
         * tensor sliced to columns [1, 3) reported strides {2, 1}, claiming its four elements
         * sit at offsets 1, 2, 3, 4 when they really sit at 1, 2, 5, 6. Nothing in this tree
         * called `slice` with `dim > 0`, which is why it survived; the first caller would
         * have read the wrong memory with no diagnostic anywhere.
         *
         * The result of an inner-axis slice is therefore **not contiguous**, and
         * `is_contiguous()` now says so. This library's operations expect contiguous
         * operands, so such a slice is a description of memory rather than something to hand
         * straight to `gemm` - which is the honest state of affairs, and a great deal better
         * than a tensor that lies about where its elements are.
         *
         * @param dim The dimension to slice (e.g., slice certain rows).
         * @param start The starting index (inclusive); negative counts back from the end.
         * @param end The ending index (exclusive); negative counts back from the end.
         * @return A new Tensor view representing the slice.
         * @throws std::runtime_error If @p dim is not a dimension of this tensor.
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
