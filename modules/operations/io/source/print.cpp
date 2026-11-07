/**************************************************************************/
/* print.cpp                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file print.cpp
 * @brief Implements IO::to_string and IO::print, which render a tensor as text.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/io/io.hpp>
#include "io_tensor_access.hpp"

namespace SushiBLAS
{
    namespace
    {
        using Detail::is_device_allocated;
        using Detail::storage_queue;

        /** @brief Returns element @p index of a host buffer of @p dtype as a (real, imag) pair. */
        std::pair<double, double> decode_element(const char* buf, Core::DataType dtype,
                                                 int64_t index)
        {
            switch (dtype)
            {
                case Core::DataType::Half:
                    return {static_cast<double>(static_cast<float>(
                                reinterpret_cast<const sycl::half*>(buf)[index])), 0.0};
                case Core::DataType::Float32:
                    return {reinterpret_cast<const float*>(buf)[index], 0.0};
                case Core::DataType::Float64:
                    return {reinterpret_cast<const double*>(buf)[index], 0.0};
                case Core::DataType::Complex32:
                {
                    const float* c = reinterpret_cast<const float*>(buf) + index * 2;
                    return {c[0], c[1]};
                }
                case Core::DataType::Complex64:
                {
                    const double* c = reinterpret_cast<const double*>(buf) + index * 2;
                    return {c[0], c[1]};
                }
                case Core::DataType::Int32:
                    return {static_cast<double>(reinterpret_cast<const int32_t*>(buf)[index]), 0.0};
                case Core::DataType::Int64:
                    return {static_cast<double>(reinterpret_cast<const int64_t*>(buf)[index]), 0.0};
                case Core::DataType::UInt32:
                    return {static_cast<double>(reinterpret_cast<const uint32_t*>(buf)[index]),
                            0.0};
                default:
                    SB_THROW_IF(true, "Cannot render a tensor of dtype {} as text.",
                                Core::to_string(dtype));
                    return {0.0, 0.0};
            }
        }

        /**
         * @brief Reads runs of a tensor's elements as (real, imaginary) pairs, through the queue
         *        for device storage.
         */
        class ElementRunReader
        {
            public:
                /**
                 * @brief Binds the reader to the tensor's first element, dtype and queue.
                 * @pre t has storage or holds no elements.
                 */
                explicit ElementRunReader(const Tensor& t)
                    : base_(static_cast<const char*>(t.data_or_null())), dtype_(t.dtype),
                      element_bytes_(Core::bytes_per_element(t.dtype)),
                      queue_(t.storage && is_device_allocated(t) ? &storage_queue(t) : nullptr)
                {

                }

                /**
                 * @brief Appends count elements to out: element first of the view, then every
                 *        stride-th element after it.
                 */
                void read(int64_t first, int64_t count, int64_t stride,
                          std::vector<std::pair<double, double>>& out)
                {
                    if (queue_ == nullptr)
                    {
                        for (int64_t i = 0; i < count; ++i)
                            out.push_back(decode_element(base_, dtype_, first + i * stride));
                        return;
                    }

                    // One copy serves a unit-stride run; any other stride copies one by one.
                    const int64_t per_copy = stride == 1 ? count : 1;
                    staging_.resize(static_cast<size_t>(per_copy) * element_bytes_);
                    for (int64_t i = 0; i < count; i += per_copy)
                    {
                        const char* source =
                            base_ + static_cast<size_t>(first + i * stride) * element_bytes_;
                        queue_->memcpy(staging_.data(), source, staging_.size()).wait();
                        for (int64_t k = 0; k < per_copy; ++k)
                            out.push_back(decode_element(staging_.data(), dtype_, k));
                    }
                }

            private:
                /// @brief The address of the view's first element, or null for an empty tensor.
                const char* base_;

                /// @brief The element type decoded.
                Core::DataType dtype_;

                /// @brief The bytes one element takes.
                size_t element_bytes_;

                /// @brief The queue device storage is copied through; null when host-readable.
                sycl::queue* queue_;

                /// @brief The host bytes of the run being decoded.
                std::vector<char> staging_;
        };

        /** @brief Holds what the text writer needs at every depth of a tensor. */
        struct TextLayout
        {
            const int64_t* shape;   /**< The size of each dimension. */
            const int64_t* strides; /**< The stride of each dimension, in elements. */
            int rank;               /**< The number of dimensions. */
            int precision;          /**< Decimal places per number. */
            int edge_items;         /**< Entries kept at each end of an elided dimension. */
        };

        /**
         * @brief Returns one element as text: a real number, or `(re+imj)` when it has an
         *        imaginary part.
         */
        std::string format_element(const std::pair<double, double>& value, int precision)
        {
            std::stringstream text;
            text << std::fixed << std::setprecision(precision);
            if (std::abs(value.second) < 1e-9)
                text << value.first;
            else
                text << "(" << value.first << (value.second >= 0 ? "+" : "") << value.second
                     << "j)";
            return text.str();
        }

        /** @brief Writes the innermost dimension starting at element offset of the view. */
        void write_innermost(std::ostream& os, ElementRunReader& reader, const TextLayout& layout,
                             int64_t offset)
        {
            const int dim = layout.rank - 1;
            const int64_t extent = layout.shape[dim];
            const int64_t stride = layout.strides[dim];
            const int64_t edge = layout.edge_items;
            const bool elided = edge >= 0 && extent > edge * 2;

            std::vector<std::pair<double, double>> values;
            if (elided)
            {
                reader.read(offset, edge, stride, values);
                reader.read(offset + (extent - edge) * stride, edge, stride, values);
            }
            else
                reader.read(offset, extent, stride, values);

            os << "[";
            for (size_t i = 0; i <= values.size(); ++i)
            {
                if (elided && i == static_cast<size_t>(edge))
                    os << "... ";
                if (i == values.size())
                    break;
                os << format_element(values[i], layout.precision)
                   << (i + 1 == values.size() ? "" : ", ");
            }
            os << "]";
        }

        /**
         * @brief Writes dimension dim and everything inside it, starting at element offset of
         *        the view.
         */
        void write_dimension(std::ostream& os, ElementRunReader& reader, const TextLayout& layout,
                             int dim, int64_t offset)
        {
            if (dim == layout.rank - 1)
            {
                write_innermost(os, reader, layout, offset);
                return;
            }

            const int64_t extent = layout.shape[dim];
            const int64_t edge = layout.edge_items;

            os << "[";
            for (int64_t i = 0; i < extent; ++i)
            {
                if (extent > edge * 2 && i == edge)
                {
                    const std::string indent(dim + 1, ' ');
                    os << "\n" << indent << "...\n" << indent;
                    i = extent - edge - 1;
                    continue;
                }

                write_dimension(os, reader, layout, dim + 1, offset + i * layout.strides[dim]);

                if (i < extent - 1)
                    os << ",\n" << std::string(dim + 1, ' ');
            }
            os << "]";

            if (dim == 0)
                os << std::endl;
        }
    } // namespace

    void IO::print(const Tensor& t, int precision, int edge_items)
    {
        std::cout << to_string(t, precision, edge_items);
    }

    std::string IO::to_string(const Tensor& t, int precision, int edge_items)
    {
        Dispatch::require_storage(t, "io.to_string", "t");

        std::stringstream ss;
        ss << "Tensor(shape=[";

        for (int i = 0; i < t.rank; ++i)
            ss << t.shape[i] << (i == t.rank - 1 ? "" : ", ");

        ss << "], dtype=";

        if (t.dtype == Core::DataType::Half)
            ss << "f16";
        else if (t.dtype == Core::DataType::Float32)
            ss << "f32";
        else if (t.dtype == Core::DataType::Float64)
            ss << "f64";
        else if (t.dtype == Core::DataType::Complex32)
            ss << "c32";
        else if (t.dtype == Core::DataType::Complex64)
            ss << "c64";
        else if (t.dtype == Core::DataType::Int32)
            ss << "i32";
        else if (t.dtype == Core::DataType::Int64)
            ss << "i64";
        else if (t.dtype == Core::DataType::UInt32)
            ss << "u32";

        ss << "):" << std::endl;
        ss << std::fixed << std::setprecision(precision);

        ElementRunReader reader(t);
        const TextLayout layout{t.shape.data(), t.strides.data(), t.rank, precision, edge_items};
        write_dimension(ss, reader, layout, 0, 0);

        return ss.str();
    }
} // namespace SushiBLAS
