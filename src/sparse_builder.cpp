/**************************************************************************/
/* sparse_builder.cpp                                                     */
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
 * @file sparse_builder.cpp
 * @brief Defines the allocation, the host transfers and the dense compression of SparseBuilder.
 * @author Mustafa Garip
 */

#include <SushiBLAS/sparse_builder.hpp>

#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Holds the bytes of a dense tensor one transfer stages on the host. */
        constexpr std::size_t K_DENSE_CHUNK_BYTES = std::size_t{4} << 20;

        /** @brief Returns whether a dense element becomes a stored entry: it is not zero. */
        template<typename T>
        [[nodiscard]] bool is_stored(const T& value) noexcept
        {
            if constexpr (Core::is_complex_v<T>)
                return value.real() != 0 || value.imag() != 0;
            else if constexpr (std::is_same_v<T, sycl::half>)
                return static_cast<float>(value) != 0.0f;
            else
                return value != T(0);
        }

        /**
         * @brief Calls visit(row, column, value) for each element of a dense matrix, in memory
         *        order.
         * @pre dense has rank 2, is contiguous and holds elements of type T.
         * @param visit Callable `(int64_t row, int64_t column, const T& value) -> void`.
         */
        template<typename T, typename Visit>
        void for_each_dense_element(const Tensor& dense, Visit visit)
        {
            const bool row_major = dense.layout == Core::Layout::RowMajor;
            const int64_t lines = row_major ? dense.shape[0] : dense.shape[1];
            const int64_t line_length = row_major ? dense.shape[1] : dense.shape[0];
            if (lines == 0 || line_length == 0)
                return;

            // A chunk holds whole lines, and at least one however long a line is.
            const int64_t line_bytes = line_length * static_cast<int64_t>(sizeof(T));
            const int64_t lines_per_chunk =
                std::max<int64_t>(1, static_cast<int64_t>(K_DENSE_CHUNK_BYTES) / line_bytes);
            std::vector<T> chunk(
                static_cast<std::size_t>(std::min(lines, lines_per_chunk) * line_length));

            sycl::queue& source_queue =
                dense.storage->get_context().get_queue(dense.storage->get_device_index());
            const T* source = static_cast<const T*>(dense.data());

            for (int64_t first = 0; first < lines; first += lines_per_chunk)
            {
                const int64_t count = std::min(lines_per_chunk, lines - first);
                source_queue.memcpy(chunk.data(), source + first * line_length,
                                    static_cast<std::size_t>(count * line_bytes)).wait();

                for (int64_t line = 0; line < count; ++line)
                {
                    for (int64_t position = 0; position < line_length; ++position)
                    {
                        const T& value =
                            chunk[static_cast<std::size_t>(line * line_length + position)];
                        if (row_major)
                            visit(first + line, position, value);
                        else
                            visit(position, first + line, value);
                    }
                }
            }
        }
    } // namespace

    void SparseBuilder::require_index_base(Core::IndexBase base)
    {
        SB_THROW_IF(!Core::is_valid(base),
                    "The index base holds the value {}, which is not an enumerator of its type.",
                    static_cast<int>(base));
    }

    void SparseBuilder::require_equal_counts(std::size_t indices, std::size_t values)
    {
        SB_THROW_IF(indices != values,
                    "A sparse vector was given {} indices and {} values; the two counts must "
                    "match.",
                    indices, values);
    }

    Tensor SparseBuilder::create_filled(int64_t elements, Core::DataType dtype,
                                        const void* source) const
    {
        const int64_t dims[1] = {elements};
        Tensor tensor = factory_->create(SushiRuntime::span<const int64_t>(dims, 1), dtype,
                                         residency_, device_index_);
        if (elements == 0)
            return tensor;

        const std::size_t bytes =
            static_cast<std::size_t>(elements) * Core::bytes_per_element(dtype);
        sycl::queue& target_queue = factory_->get_context().get_queue(device_index_);
        target_queue.memcpy(tensor.data(), source, bytes).wait();
        return tensor;
    }

    SparseMatrix SparseBuilder::create_matrix(int64_t rows, int64_t cols, Core::IndexBase base,
                                              Core::DataType index_dtype,
                                              Core::DataType value_dtype, const void* row_offsets,
                                              const void* column_indices, const void* values,
                                              int64_t nnz) const
    {
        SparseMatrix matrix;
        matrix.rows = rows;
        matrix.cols = cols;
        matrix.index_base = base;
        matrix.row_offsets = create_filled(rows + 1, index_dtype, row_offsets);
        matrix.column_indices = create_filled(nnz, index_dtype, column_indices);
        matrix.values = create_filled(nnz, value_dtype, values);
        return matrix;
    }

    SparseVector SparseBuilder::create_vector(int64_t size, Core::IndexBase base,
                                              Core::DataType index_dtype,
                                              Core::DataType value_dtype, const void* indices,
                                              const void* values, int64_t nnz) const
    {
        SparseVector vector;
        vector.size = size;
        vector.index_base = base;
        vector.indices = create_filled(nnz, index_dtype, indices);
        vector.values = create_filled(nnz, value_dtype, values);
        return vector;
    }

    template<typename T, typename Index>
    SparseMatrix SparseBuilder::compress_dense(const Tensor& dense, Core::IndexBase base) const
    {
        const int64_t rows = dense.shape[0];
        const int64_t cols = dense.shape[1];
        const int64_t first_index = Core::first_index(base);

        // After the first pass cursor[r + 1] counts the stored entries of row r.
        std::vector<int64_t> cursor(static_cast<std::size_t>(rows) + 1, 0);
        for_each_dense_element<T>(dense, [&cursor](int64_t row, int64_t, const T& value)
        {
            if (is_stored(value))
                ++cursor[static_cast<std::size_t>(row) + 1];
        });
        for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r)
            cursor[r + 1] += cursor[r];

        const int64_t nnz = cursor[static_cast<std::size_t>(rows)];
        Sparse::require_index_capacity<Index>(nnz + first_index, "entry count");
        Sparse::require_index_capacity<Index>(cols - 1 + first_index, "last column");

        Sparse::HostCsr<Index, T> csr;
        csr.row_offsets.resize(static_cast<std::size_t>(rows) + 1);
        for (std::size_t r = 0; r <= static_cast<std::size_t>(rows); ++r)
            csr.row_offsets[r] = static_cast<Index>(cursor[r] + first_index);
        csr.column_indices.resize(static_cast<std::size_t>(nnz));
        csr.values.resize(static_cast<std::size_t>(nnz));

        // Both layouts visit a row's elements by ascending column, so each row comes out sorted.
        for_each_dense_element<T>(dense, [&cursor, &csr, first_index](int64_t row, int64_t column,
                                                                      const T& value)
        {
            if (!is_stored(value))
                return;

            const std::size_t slot =
                static_cast<std::size_t>(cursor[static_cast<std::size_t>(row)]++);
            csr.column_indices[slot] = static_cast<Index>(column + first_index);
            csr.values[slot] = value;
        });

        return create_matrix(rows, cols, base, Detail::HostDataType<Index>::K_VALUE, dense.dtype,
                             csr.row_offsets.data(), csr.column_indices.data(), csr.values.data(),
                             nnz);
    }

    SparseMatrix SparseBuilder::matrix_from_dense(const Tensor& dense, Core::DataType index_dtype,
                                                  Core::IndexBase base) const
    {
        require_index_base(base);
        SB_THROW_IF(dense.rank != 2, "A dense matrix has rank 2; this tensor has rank {}.",
                    dense.rank);
        SB_THROW_IF(dense.num_elements > 0 && !dense.storage,
                    "The dense tensor holds {} elements but has no storage.", dense.num_elements);
        SB_THROW_IF(!dense.is_contiguous(),
                    "The dense tensor is a strided view; it must be contiguous.");
        SB_THROW_IF(!Core::is_integer_dtype(index_dtype),
                    "A sparse matrix cannot index with {}; its index dtype is INT32 or INT64.",
                    Core::to_string(index_dtype));

        const bool narrow = index_dtype == Core::DataType::Int32;
        switch (dense.dtype)
        {
            case Core::DataType::Half:
                return narrow ? compress_dense<sycl::half, int32_t>(dense, base)
                              : compress_dense<sycl::half, int64_t>(dense, base);
            case Core::DataType::Float32:
                return narrow ? compress_dense<float, int32_t>(dense, base)
                              : compress_dense<float, int64_t>(dense, base);
            case Core::DataType::Float64:
                return narrow ? compress_dense<double, int32_t>(dense, base)
                              : compress_dense<double, int64_t>(dense, base);
            case Core::DataType::Complex32:
                return narrow ? compress_dense<std::complex<float>, int32_t>(dense, base)
                              : compress_dense<std::complex<float>, int64_t>(dense, base);
            case Core::DataType::Complex64:
                return narrow ? compress_dense<std::complex<double>, int32_t>(dense, base)
                              : compress_dense<std::complex<double>, int64_t>(dense, base);
            case Core::DataType::Int32:
            case Core::DataType::Int64:
                break;
        }

        SB_THROW_IF(true,
                    "A sparse matrix cannot hold {} values; its dtype is floating-point or "
                    "complex.",
                    Core::to_string(dense.dtype));
        return SparseMatrix();
    }
} // namespace SushiBLAS
