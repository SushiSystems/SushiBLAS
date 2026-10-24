/**************************************************************************/
/* io.hpp                                                                 */
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
 * @file io.hpp
 * @brief Declares IO, which saves, loads and displays tensors.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/host_data_type.hpp>
#include <SushiBLAS/io/io_status.hpp>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    /**
     * @class IO
     * @brief Saves, loads and displays tensors in .sushi, raw binary and NumPy .npy form.
     *
     * Every call reads or writes what the tensor's storage holds when it is called; the caller
     * runs the graph that writes a tensor before it reads the tensor here.
     */
    class IO
    {
        public:
            /**
             * @brief Saves a tensor in the native .sushi format: a header, then the raw data.
             * @param t The tensor to save; stored and contiguous.
             * @return FileUnwritable when a write fails, after the partial file is removed.
             * @throws std::invalid_argument when t is strided or has no storage.
             */
            [[nodiscard]] IOStatus save(const Tensor& t, const std::string& path);

            /**
             * @brief Loads a .sushi file into a tensor.
             * @param t A stored, contiguous tensor; the file's rank, shape, dtype and layout must
             *        match it.
             * @return The mismatch or the file defect found; t is unchanged unless the status is
             *         FileUnreadable.
             * @throws std::invalid_argument when t is strided or has no storage.
             */
            [[nodiscard]] IOStatus load(Tensor& t, const std::string& path);

            /**
             * @brief Copies the tensor's elements into host memory.
             * @param t A stored, contiguous tensor; any residency.
             * @param bytes Must equal num_elements times the dtype's element size.
             * @throws std::invalid_argument when t breaks those rules or bytes differs.
             */
            void read(const Tensor& t, void* dst, size_t bytes);

            /**
             * @brief Copies host memory into the tensor's elements.
             * @param t A stored, contiguous tensor; any residency.
             * @param bytes Must equal num_elements times the dtype's element size.
             * @throws std::invalid_argument when t breaks those rules or bytes differs.
             */
            void write(Tensor& t, const void* src, size_t bytes);

            /**
             * @brief Returns the tensor's elements as a host vector.
             * @tparam T The host type matching t.dtype (see Detail::HostDataType).
             * @throws std::invalid_argument when t's dtype is not T's, or read() rejects t.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> to_vector(const Tensor& t)
            {
                require_element_type(t, Detail::HostDataType<T>::K_VALUE, "io.to_vector");

                std::vector<T> out(static_cast<size_t>(t.num_elements));
                read(t, out.data(), out.size() * sizeof(T));
                return out;
            }

            /**
             * @brief Copies a host vector into the tensor's elements.
             * @tparam T The host type matching t.dtype (see Detail::HostDataType).
             * @throws std::invalid_argument when t's dtype is not T's, the sizes differ, or
             * write() rejects t.
             */
            template<typename T>
            void from_vector(Tensor& t, const std::vector<T>& values)
            {
                require_element_type(t, Detail::HostDataType<T>::K_VALUE, "io.from_vector");
                write(t, values.data(), values.size() * sizeof(T));
            }

            /**
             * @brief Saves a tensor in NumPy .npy format, version 1.0.
             * @param t Stored, contiguous, and not ColumnMajor at rank 3 or more.
             * @return FileUnwritable when a write fails, after the partial file is removed.
             * @throws std::invalid_argument when t breaks one of those rules.
             */
            [[nodiscard]] IOStatus save_npy(const Tensor& t, const std::string& path);

            /**
             * @brief Loads a .npy file of version 1.0, 2.0 or 3.0.
             * @param t A stored, contiguous tensor, not ColumnMajor at rank 3 or more; the
             *        header's descr, fortran_order and shape must match it.
             * @return MalformedFile on another version or a malformed header, or the mismatch.
             * @throws std::invalid_argument when t breaks one of those rules.
             * @see modules/README.md
             */
            [[nodiscard]] IOStatus load_npy(Tensor& t, const std::string& path);

            /**
             * @brief Saves the tensor's raw bytes, with no header.
             * @param t The tensor to save; stored and contiguous.
             * @return FileUnwritable when a write fails, after the partial file is removed.
             * @throws std::invalid_argument when t is strided or has no storage.
             */
            [[nodiscard]] IOStatus save_bin(const Tensor& t, const std::string& path);

            /**
             * @brief Loads raw bytes with no header.
             * @param t The target tensor; stored and contiguous.
             * @return Truncated on a short file, before any element changes; FileUnreadable on a
             *         failed read, after some have.
             * @throws std::invalid_argument when t is strided or has no storage.
             */
            [[nodiscard]] IOStatus load_bin(Tensor& t, const std::string& path);

            /**
             * @brief Print tensor content to the console in a readable format.
             * @param t The tensor to print.
             * @param precision Number of decimal places for numeric display.
             * @param edge_items Number of items to show at the edges of each dimension.
             */
            void print(const Tensor& t, int precision = 4, int edge_items = 3);

            /**
             * @brief Returns the tensor's content as a string in the format print() uses.
             * @param t Any stored view; its elements are read through its strides.
             * @param precision Number of decimal places for numeric display.
             * @param edge_items Number of items shown at the edges of each dimension.
             */
            std::string to_string(const Tensor& t, int precision = 4, int edge_items = 3);

        private:
            /**
             * @brief Throws unless t carries the data type a host element type maps to.
             * @throws std::invalid_argument naming op on a mismatch.
             */
            static void require_element_type(const Tensor& t, Core::DataType expected,
                                             const char* op);
    };

} // namespace SushiBLAS
