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
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    class Engine;

    /** @brief Holds the host-type traits IO's templates use. */
    namespace Detail
    {
        /**
         * @brief Maps a host element type to the DataType a tensor of it carries.
         * @tparam T One of float, double, sycl::half, std::complex<float>,
         * std::complex<double>, int32_t, int64_t.
         */
        template<typename T> struct HostDataType;

        /** @brief Maps float to FLOAT32. */
        template<> struct HostDataType<float>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Float32;
        };

        /** @brief Maps double to FLOAT64. */
        template<> struct HostDataType<double>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Float64;
        };

        /** @brief Maps sycl::half to HALF. */
        template<> struct HostDataType<sycl::half>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Half;
        };

        /** @brief Maps std::complex<float> to COMPLEX32. */
        template<> struct HostDataType<std::complex<float>>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Complex32;
        };

        /** @brief Maps std::complex<double> to COMPLEX64. */
        template<> struct HostDataType<std::complex<double>>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Complex64;
        };

        /** @brief Maps int32_t to INT32. */
        template<> struct HostDataType<int32_t>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Int32;
        };

        /** @brief Maps int64_t to INT64. */
        template<> struct HostDataType<int64_t>
        {
            /** @brief The matching DataType. */
            static constexpr Core::DataType K_VALUE = Core::DataType::Int64;
        };
    } // namespace Detail

    /**
     * @class IO
     * @brief Input/Output operations for Tensors.
     *
     * This class provides methods to save, load, and display tensors. It supports
     * both native .sushi formats and NumPy .npy exports for interoperability.
     */
    class IO
    {
        public:
            /**
             * @brief Construct a new IO object.
             * @param e Reference to the parent Engine.
             */
            explicit IO(Engine& e) : engine_(e)
            {

            }

            /**
             * @brief Saves a tensor in the native .sushi format: a header, then the raw data.
             * @param t The tensor to save; stored and contiguous.
             * @throws std::invalid_argument when t is strided or has no storage.
             * @throws std::runtime_error when a write fails; the partial file is removed.
             */
            void save(const Tensor& t, const std::string& path);

            /**
             * @brief Load a .sushi file into a tensor.
             *
             * Runs the recorded graph first, then verifies that the file metadata matches the
             * tensor.
             * @param t The target tensor to load data into.
             * @param path The filesystem path of the .sushi file.
             */
            void load(Tensor& t, const std::string& path);

            /**
             * @brief Runs the recorded graph, then copies the tensor's elements into host memory.
             * @param t A stored, contiguous tensor; any residency.
             * @param bytes Must equal num_elements times the dtype's element size.
             * @throws std::invalid_argument when t breaks those rules or bytes differs.
             */
            void read(const Tensor& t, void* dst, size_t bytes);

            /**
             * @brief Runs the recorded graph, then copies host memory into the tensor's elements.
             * @param t A stored, contiguous tensor; any residency.
             * @param bytes Must equal num_elements times the dtype's element size.
             * @throws std::invalid_argument when t breaks those rules or bytes differs.
             */
            void write(Tensor& t, const void* src, size_t bytes);

            /**
             * @brief Returns the tensor's elements as a host vector, after running the graph.
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
             * @brief Copies a host vector into the tensor's elements, after running the graph.
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
             * @param t Stored, contiguous, and not COLUMN_MAJOR at rank 3 or more.
             * @throws std::invalid_argument when t breaks one of those rules.
             * @throws std::runtime_error when a write fails; the partial file is removed.
             */
            void save_npy(const Tensor& t, const std::string& path);

            /**
             * @brief Runs the recorded graph, then loads a .npy file of version 1.0, 2.0 or 3.0.
             * @param t A stored, contiguous tensor, not COLUMN_MAJOR at rank 3 or more; the
             *        header's descr, fortran_order and shape must match it.
             * @throws std::invalid_argument when t breaks one of those rules.
             * @throws std::runtime_error on another version, a malformed header or a mismatch.
             * @see include/SushiBLAS/README.md
             */
            void load_npy(Tensor& t, const std::string& path);

            /**
             * @brief Raw binary save (no header).
             * @param t The tensor to save; stored and contiguous.
             * @param path The filesystem path for the binary file.
             * @throws std::runtime_error when a write fails; the partial file is removed.
             */
            void save_bin(const Tensor& t, const std::string& path);

            /**
             * @brief Raw binary load (no header), after running the recorded graph.
             * @param t The target tensor to load data into.
             * @param path The filesystem path of the binary file.
             * @throws std::runtime_error on a short file before any element changes, and on a
             *         failed read after some have.
             */
            void load_bin(Tensor& t, const std::string& path);

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

            /// @brief The engine whose recorded graph runs before every transfer.
            Engine& engine_;
    };

} // namespace SushiBLAS
