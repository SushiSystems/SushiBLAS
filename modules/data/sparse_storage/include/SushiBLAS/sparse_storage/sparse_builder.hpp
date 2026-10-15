/**************************************************************************/
/* sparse_builder.hpp                                                     */
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
 * @file sparse_builder.hpp
 * @brief Declares SparseBuilder, which builds sparse vectors and CSR matrices from host data.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor/host_data_type.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/sparse_assembly/sparse_assembly.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    /**
     * @brief Builds sparse operands on one device of one tensor factory, from data on the host.
     * @see modules/README.md
     */
    class SparseBuilder final
    {
        public:
            /**
             * @brief Binds the builder to the factory that allocates and to where its tensors live.
             * @param factory Allocates every tensor; it must outlive the builder.
             * @param device_index Device queue that owns every allocation.
             */
            explicit SparseBuilder(
                TensorFactory& factory,
                SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                std::size_t device_index = 0) noexcept
                : factory_(&factory), residency_(residency), device_index_(device_index)
            {

            }

            SparseBuilder(const SparseBuilder&) = delete;
            SparseBuilder& operator=(const SparseBuilder&) = delete;
            SparseBuilder(SparseBuilder&&) = delete;
            SparseBuilder& operator=(SparseBuilder&&) = delete;
            ~SparseBuilder() = default;

            /**
             * @brief Builds a CSR matrix from COO triplets, adding the values of repeated pairs.
             * @tparam Index int32_t or int64_t; it becomes the index dtype of the matrix.
             * @tparam T The host type of a floating-point or complex dtype.
             * @return A matrix that holds each (row, column) pair once, by ascending column in a
             *         row.
             * @throws std::runtime_error as Sparse::assemble_csr and TensorFactory::create state.
             */
            template<typename Index, typename T>
            [[nodiscard]] SparseMatrix matrix_from_triplets(
                int64_t rows, int64_t cols, const std::vector<Index>& row_indices,
                const std::vector<Index>& column_indices, const std::vector<T>& values,
                Core::IndexBase base = Core::IndexBase::Zero) const
            {
                static_assert(!std::is_integral_v<T>,
                              "A sparse value is floating-point or complex.");
                require_index_base(base);

                const Sparse::HostCsr<Index, T> csr = Sparse::assemble_csr(
                    rows, cols, Core::first_index(base), row_indices, column_indices, values);
                return create_matrix(rows, cols, base, Detail::HostDataType<Index>::K_VALUE,
                                     Detail::HostDataType<T>::K_VALUE, csr.row_offsets.data(),
                                     csr.column_indices.data(), csr.values.data(),
                                     static_cast<int64_t>(csr.values.size()));
            }

            /**
             * @brief Builds a CSR matrix from the non-zero elements of a dense rank-2 tensor.
             * @pre No recorded task that writes dense is waiting to run.
             * @param dense Contiguous, of a floating-point or complex dtype.
             * @return A matrix that stores every element not equal to zero, NaN included.
             * @throws std::runtime_error on an operand outside those rules, an index_dtype that is
             *         not INT32 or INT64 or is too narrow, or as TensorFactory::create states.
             */
            [[nodiscard]] SparseMatrix matrix_from_dense(
                const Tensor& dense, Core::DataType index_dtype = Core::DataType::Int64,
                Core::IndexBase base = Core::IndexBase::Zero) const;

            /**
             * @brief Builds a sparse vector from its stored positions and values.
             * @tparam Index int32_t or int64_t; it becomes the index dtype of the vector.
             * @tparam T The host type of the value dtype.
             * @param size The length of the dense vector the result compresses.
             * @throws std::runtime_error on arrays of different lengths, and as
             *         Sparse::require_distinct_positions and TensorFactory::create state.
             */
            template<typename Index, typename T>
            [[nodiscard]] SparseVector vector_from_entries(
                int64_t size, const std::vector<Index>& indices, const std::vector<T>& values,
                Core::IndexBase base = Core::IndexBase::Zero) const
            {
                require_index_base(base);
                require_equal_counts(indices.size(), values.size());
                Sparse::require_distinct_positions(size, Core::first_index(base), indices);

                return create_vector(size, base, Detail::HostDataType<Index>::K_VALUE,
                                     Detail::HostDataType<T>::K_VALUE, indices.data(),
                                     values.data(), static_cast<int64_t>(values.size()));
            }

        private:
            /** @brief Throws std::runtime_error unless base is an enumerator of Core::IndexBase. */
            static void require_index_base(Core::IndexBase base);

            /** @brief Throws std::runtime_error unless a vector has as many indices as values. */
            static void require_equal_counts(std::size_t indices, std::size_t values);

            /**
             * @brief Allocates a rank-1 tensor of elements and fills it from host memory.
             * @param source elements values of dtype; never read when elements is 0.
             * @return A tensor without storage when elements is 0.
             */
            [[nodiscard]] Tensor create_filled(int64_t elements, Core::DataType dtype,
                                               const void* source) const;

            /** @brief Allocates the three tensors of a CSR matrix and fills them from the host. */
            [[nodiscard]] SparseMatrix create_matrix(int64_t rows, int64_t cols,
                                                     Core::IndexBase base,
                                                     Core::DataType index_dtype,
                                                     Core::DataType value_dtype,
                                                     const void* row_offsets,
                                                     const void* column_indices, const void* values,
                                                     int64_t nnz) const;

            /** @brief Allocates the two tensors of a sparse vector and fills them from the host. */
            [[nodiscard]] SparseVector create_vector(int64_t size, Core::IndexBase base,
                                                     Core::DataType index_dtype,
                                                     Core::DataType value_dtype,
                                                     const void* indices, const void* values,
                                                     int64_t nnz) const;

            /**
             * @brief Reads a dense tensor of element type T chunk by chunk and compresses it.
             * @tparam Index int32_t or int64_t.
             */
            template<typename T, typename Index>
            [[nodiscard]] SparseMatrix compress_dense(const Tensor& dense,
                                                      Core::IndexBase base) const;

            /// @brief The factory that allocates; never null.
            TensorFactory* factory_;

            /// @brief The kind of USM every built tensor lives in.
            SushiRuntime::Residency residency_;

            /// @brief The device queue that owns every built tensor.
            std::size_t device_index_;
    };
} // namespace SushiBLAS
