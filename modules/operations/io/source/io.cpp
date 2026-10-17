/**************************************************************************/
/* io.cpp                                                                 */
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
 * @file io.cpp
 * @brief Defines IO, which saves and loads tensors as .sushi, raw binary and .npy files and
 * renders them as text.
 * @author Mustafa Garip
 */

#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/io/io_log_category.hpp>
#include <SushiRuntime/core/text_format.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <SushiBLAS/io/io.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Holds the only .sushi format version this build reads and writes. */
        constexpr uint32_t K_SUSHI_FORMAT_VERSION = 1;

        /** @brief Carries the status of a failed file operation from its site to the IO call. */
        struct IOFailure
        {
            IOStatus status; /**< Why the operation failed. */
        };

        /** @brief Logs why a file operation failed, once, and leaves through an IOFailure. */
        template<typename... Args>
        [[noreturn]] void fail_io(IOStatus status, [[maybe_unused]] std::string_view format,
                                  [[maybe_unused]] const Args&... args)
        {
            SB_LOG_ERROR(K_IO_LOG_CATEGORY, "io: {}", SushiRuntime::Detail::sr_format(format, args...));
            throw IOFailure{status};
        }

        /**
         * @brief Runs the tasks recorded on the engine, so a host transfer sees what they wrote.
         * @throws std::runtime_error with the runtime's message if the run fails.
         */
        void run_recorded(Engine& engine)
        {
            const SushiRuntime::Result<void> ran = engine.execute();
            SB_THROW_IF(!ran, "io: the recorded tasks failed before the host transfer: {}",
                        ran.error().message);
        }

        /**
         * @brief Runs the tasks recorded on the engine, so a file operation sees what they wrote.
         * @return Ok, or RunFailed after the runtime's message is logged once.
         */
        [[nodiscard]] IOStatus run_before_file(Engine& engine)
        {
            const SushiRuntime::Result<void> ran = engine.execute();
            if (ran)
                return IOStatus::Ok;

            SB_LOG_ERROR(K_IO_LOG_CATEGORY, "io: the recorded tasks failed before the file operation: {}",
                         ran.error().message);
            return IOStatus::RunFailed;
        }

        /** @brief Runs file work and returns Ok, or the status of the IOFailure that ended it. */
        template<typename FileWork>
        [[nodiscard]] IOStatus status_of(FileWork file_work)
        {
            try
            {
                file_work();
            }
            catch (const IOFailure& failure)
            {
                return failure.status;
            }
            return IOStatus::Ok;
        }

        /** @brief Returns the number of bytes the tensor's elements occupy. */
        size_t payload_bytes(const Tensor& t)
        {
            return static_cast<size_t>(t.num_elements) * Core::bytes_per_element(t.dtype);
        }

        /** @brief Reports whether the host reaches the tensor's storage only through the queue. */
        bool is_device_allocated(const Tensor& t)
        {
            return t.storage->residency() == SushiRuntime::Residency::Device;
        }

        /**
         * @brief Returns the queue of the device that holds the tensor's storage.
         * @pre t has storage.
         */
        sycl::queue& storage_queue(const Tensor& t)
        {
            return t.storage->queue();
        }

        /** @brief Holds the fields of a .npy header dictionary. */
        struct NpyHeader
        {
            std::string descr;          /**< The dtype descriptor, such as "<f4". */
            bool fortran_order = false; /**< Whether the payload is in Fortran order. */
            std::vector<int64_t> shape; /**< The dimensions, outermost first. */
        };

        /**
         * @brief Returns the .npy descriptor of a dtype.
         * @throws std::runtime_error for a dtype with no descriptor.
         */
        const char* npy_descr(Core::DataType dtype)
        {
            switch (dtype)
            {
                case Core::DataType::Float32:   return "<f4";
                case Core::DataType::Float64:   return "<f8";
                case Core::DataType::Complex32: return "<c8";
                case Core::DataType::Complex64: return "<c16";
                case Core::DataType::Half:      return "<f2";
                case Core::DataType::Int32:     return "<i4";
                case Core::DataType::Int64:     return "<i8";
                default:
                    SB_THROW_IF(true, "Dtype {} has no .npy descriptor.", Core::to_string(dtype));
                    return "";
            }
        }

        /** @brief Reads a .npy header dictionary, a Python literal, one token at a time. */
        class NpyHeaderParser
        {
            public:
                /** @brief Binds the parser to the header text and the path its errors name. */
                NpyHeaderParser(const std::string& text, const std::string& path)
                    : text_(text), path_(path)
                {

                }

                /**
                 * @brief Returns the parsed header.
                 * @throws IOFailure with MalformedFile on malformed text or a wrong set of keys.
                 */
                NpyHeader parse()
                {
                    NpyHeader header;
                    bool has_descr = false;
                    bool has_order = false;
                    bool has_shape = false;

                    expect('{');
                    while (true)
                    {
                        skip_space();
                        if (peek() == '}')
                            break;

                        const std::string key = read_quoted();
                        expect(':');

                        if (key == "descr")
                        {
                            claim_key(has_descr, key);
                            header.descr = read_quoted();
                        }
                        else if (key == "fortran_order")
                        {
                            claim_key(has_order, key);
                            header.fortran_order = read_bool();
                        }
                        else if (key == "shape")
                        {
                            claim_key(has_shape, key);
                            header.shape = read_shape();
                        }
                        else
                            fail("unknown key '" + key + "'");

                        skip_space();
                        if (peek() == ',')
                            ++pos_;
                        else if (peek() != '}')
                            fail("expected ',' or '}' after the value of '" + key + "'");
                    }

                    if (!has_descr || !has_order || !has_shape)
                        fail("descr, fortran_order and shape are all required");
                    return header;
                }

            private:
                /** @brief Fails with MalformedFile, naming the file and the defect. */
                [[noreturn]] void fail(const std::string& what) const
                {
                    fail_io(IOStatus::MalformedFile,
                            "The .npy header of {} is malformed at byte {}: {}.", path_, pos_,
                            what);
                }

                /** @brief Marks a key as seen and fails when it was seen before. */
                void claim_key(bool& seen, const std::string& key) const
                {
                    if (seen)
                        fail("duplicate key '" + key + "'");
                    seen = true;
                }

                /** @brief Advances past spaces, tabs and newlines. */
                void skip_space()
                {
                    while (pos_ < text_.size() &&
                           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n'))
                        ++pos_;
                }

                /** @brief Returns the current character, or '\0' at the end. */
                char peek() const
                {
                    return pos_ < text_.size() ? text_[pos_] : '\0';
                }

                /** @brief Skips space and consumes @p c; fails when another character is there. */
                void expect(char c)
                {
                    skip_space();
                    if (peek() != c)
                        fail(std::string("expected '") + c + "'");
                    ++pos_;
                }

                /** @brief Returns the content of a single- or double-quoted string. */
                std::string read_quoted()
                {
                    skip_space();
                    const char quote = peek();
                    if (quote != '\'' && quote != '"')
                        fail("expected a quoted string");

                    const size_t end = text_.find(quote, pos_ + 1);
                    if (end == std::string::npos)
                        fail("unterminated string");

                    std::string value = text_.substr(pos_ + 1, end - pos_ - 1);
                    pos_ = end + 1;
                    return value;
                }

                /** @brief Returns the value of a True or False literal. */
                bool read_bool()
                {
                    skip_space();
                    if (text_.compare(pos_, 4, "True") == 0)
                    {
                        pos_ += 4;
                        return true;
                    }
                    if (text_.compare(pos_, 5, "False") == 0)
                    {
                        pos_ += 5;
                        return false;
                    }
                    fail("expected True or False");
                }

                /** @brief Returns the dimensions of a tuple of non-negative integers. */
                std::vector<int64_t> read_shape()
                {
                    std::vector<int64_t> dims;
                    expect('(');
                    while (true)
                    {
                        skip_space();
                        if (peek() == ')')
                        {
                            ++pos_;
                            return dims;
                        }

                        dims.push_back(read_dimension());

                        skip_space();
                        if (peek() == ',')
                            ++pos_;
                        else if (peek() != ')')
                            fail("expected ',' or ')' in shape");
                    }
                }

                /** @brief Returns one non-negative decimal integer. */
                int64_t read_dimension()
                {
                    if (peek() < '0' || peek() > '9')
                        fail("expected a non-negative integer in shape");

                    int64_t value = 0;
                    while (peek() >= '0' && peek() <= '9')
                    {
                        const int64_t digit = peek() - '0';
                        if (value > (std::numeric_limits<int64_t>::max() - digit) / 10)
                            fail("dimension overflows int64");
                        value = value * 10 + digit;
                        ++pos_;
                    }
                    return value;
                }

                /// @brief The header text.
                const std::string& text_;

                /// @brief The path named in error messages.
                const std::string& path_;

                /// @brief The index of the next unread character.
                size_t pos_ = 0;
        };

        /**
         * @brief Fails unless a parsed .npy header gives exactly the tensor's dtype, order, shape.
         * @throws IOFailure with the mismatch of the first field that differs.
         */
        void require_npy_header_matches(const NpyHeader& header, const Tensor& t,
                                        const std::string& path)
        {
            const std::string expected_descr = npy_descr(t.dtype);
            if (header.descr != expected_descr)
                fail_io(IOStatus::DtypeMismatch,
                        "File {} holds .npy dtype '{}'; the tensor it is loaded into needs '{}'.",
                        path, header.descr, expected_descr);

            const bool column_major = t.layout == Core::Layout::ColumnMajor;

            // At rank 0 and 1 C and Fortran order lay the elements out identically.
            if (t.rank >= 2 && header.fortran_order != column_major)
                fail_io(IOStatus::LayoutMismatch,
                        "File {} has fortran_order {}; the tensor it is loaded into is {}.", path,
                        header.fortran_order ? "True" : "False",
                        column_major ? "ColumnMajor" : "RowMajor");

            bool same_shape = header.shape.size() == static_cast<size_t>(t.rank);
            for (int i = 0; same_shape && i < t.rank; ++i)
                same_shape = header.shape[i] == t.shape[i];
            if (!same_shape)
                fail_io(IOStatus::ShapeMismatch,
                        "File {} holds a .npy array of rank {} whose shape differs from the "
                        "tensor's (rank {}).", path, header.shape.size(), t.rank);
        }

        /** @brief Reads exactly @p bytes into @p destination; fails Truncated on a short file. */
        void read_exact(std::ifstream& ifs, char* destination, size_t bytes,
                        const char* part, const std::string& path)
        {
            ifs.read(destination, static_cast<std::streamsize>(bytes));

            const size_t bytes_read = static_cast<size_t>(ifs.gcount());
            if (bytes_read != bytes)
                fail_io(IOStatus::Truncated,
                        "File {} ends inside its {}: {} bytes expected, {} read; nothing was "
                        "loaded.",
                        path, part, bytes, bytes_read);
        }

        /** @brief Writes @p bytes from @p source; fails FileUnwritable on a stream failure. */
        void write_checked(std::ofstream& ofs, const void* source, size_t bytes, const char* part,
                           const std::string& path)
        {
            ofs.write(static_cast<const char*>(source), static_cast<std::streamsize>(bytes));
            if (!ofs)
                fail_io(IOStatus::FileUnwritable,
                        "Writing the {} of {} failed after {} bytes were requested.", part, path,
                        bytes);
        }

        /** @brief Holds the bytes one file transfer stages on the host for device storage. */
        constexpr size_t K_IO_CHUNK_BYTES = size_t{4} << 20;

        /**
         * @brief Returns the bytes left in the stream after its read position.
         * @throws IOFailure with FileUnreadable if the stream cannot seek.
         */
        std::uintmax_t remaining_bytes(std::ifstream& ifs, const std::string& path)
        {
            const std::streampos here = ifs.tellg();
            ifs.seekg(0, std::ios::end);
            const std::streampos end = ifs.tellg();
            ifs.seekg(here);
            if (!ifs || here == std::streampos(-1) || end < here)
                fail_io(IOStatus::FileUnreadable, "The length of file {} cannot be measured.",
                        path);
            return static_cast<std::uintmax_t>(end - here);
        }

        /**
         * @brief Reads bytes of tensor data into destination.
         * @throws IOFailure with FileUnreadable if the stream fails; destination then holds a
         *         partial load.
         */
        void read_tensor_bytes(std::ifstream& ifs, char* destination, size_t bytes,
                               const std::string& path)
        {
            ifs.read(destination, static_cast<std::streamsize>(bytes));

            const size_t bytes_read = static_cast<size_t>(ifs.gcount());
            if (bytes_read != bytes)
                fail_io(IOStatus::FileUnreadable,
                        "Reading the tensor data of {} failed after {} of {} bytes; the tensor "
                        "holds a partial load.",
                        path, bytes_read, bytes);
        }

        /**
         * @brief Writes the tensor's bytes to the stream, staging device storage one chunk at a
         *        time.
         */
        void write_payload(std::ofstream& ofs, const Tensor& t, const std::string& path)
        {
            const size_t bytes = payload_bytes(t);
            if (bytes == 0)
                return;

            const char* source = static_cast<const char*>(t.data());
            if (!is_device_allocated(t))
            {
                write_checked(ofs, source, bytes, "tensor data", path);
                return;
            }

            const size_t chunk_bytes = std::min(bytes, K_IO_CHUNK_BYTES);
            const std::unique_ptr<char[]> chunk(new char[chunk_bytes]);
            sycl::queue& queue = storage_queue(t);
            for (size_t done = 0; done < bytes; done += chunk_bytes)
            {
                const size_t step = std::min(chunk_bytes, bytes - done);
                queue.memcpy(chunk.get(), source + done, step).wait();
                write_checked(ofs, chunk.get(), step, "tensor data", path);
            }
        }

        /**
         * @brief Reads the tensor's bytes from the stream, straight into host-readable storage
         *        and one chunk at a time into device storage.
         * @throws IOFailure with Truncated before any element changes if the file is too short.
         */
        void read_payload(std::ifstream& ifs, Tensor& t, const std::string& path)
        {
            const size_t bytes = payload_bytes(t);
            if (bytes == 0)
                return;

            const std::uintmax_t available = remaining_bytes(ifs, path);
            if (available < bytes)
                fail_io(IOStatus::Truncated,
                        "File {} ends inside its tensor data: {} bytes expected, {} present; "
                        "nothing was loaded.",
                        path, bytes, available);

            char* destination = static_cast<char*>(t.data());
            if (!is_device_allocated(t))
            {
                read_tensor_bytes(ifs, destination, bytes, path);
                return;
            }

            const size_t chunk_bytes = std::min(bytes, K_IO_CHUNK_BYTES);
            const std::unique_ptr<char[]> chunk(new char[chunk_bytes]);
            sycl::queue& queue = storage_queue(t);
            for (size_t done = 0; done < bytes; done += chunk_bytes)
            {
                const size_t step = std::min(chunk_bytes, bytes - done);
                read_tensor_bytes(ifs, chunk.get(), step, path);
                queue.memcpy(destination + done, chunk.get(), step).wait();
            }
        }

        /**
         * @brief Creates the file at @p path, runs @p write_body on it, deletes it on failure.
         * @throws IOFailure with FileUnwritable when the open, a write or the close fails.
         */
        template<typename WriteBody>
        void write_file(const std::string& path, WriteBody write_body)
        {
            std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
            if (!ofs.is_open())
                fail_io(IOStatus::FileUnwritable, "Failed to open file for writing: {}", path);

            try
            {
                write_body(ofs);
                ofs.close();
                if (ofs.fail())
                    fail_io(IOStatus::FileUnwritable,
                            "Closing {} failed; the file may be incomplete.", path);
            }
            catch (...)
            {
                if (ofs.is_open())
                    ofs.close();
                std::remove(path.c_str());
                throw;
            }
        }

        /** @brief Throws unless the tensor is a dense, stored view whose bytes copy as one run. */
        void require_dense_payload(const Tensor& t, const char* op)
        {
            Dispatch::require_storage(t, op, "t");
            Dispatch::require_contiguous(t, op, "t");
        }

        /**
         * @brief Throws unless .npy can describe the tensor's memory order.
         * @throws std::invalid_argument naming op for a COLUMN_MAJOR tensor of rank 3 or more.
         */
        void require_npy_layout(const Tensor& t, const char* op)
        {
            // Batch axes of a COLUMN_MAJOR tensor are outermost (tensor.hpp), not Fortran order.
            if (t.layout == Core::Layout::ColumnMajor && t.rank >= 3)
                Dispatch::Detail::throw_operand_error(
                    op, "t",
                    "is COLUMN_MAJOR with rank {}; its batch axes are outermost, so its memory is "
                    "neither C nor Fortran order and .npy cannot describe it",
                    t.rank);
        }

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

        /** @brief Holds the largest .npy header dictionary this build reads, in bytes. */
        constexpr size_t K_MAX_NPY_HEADER_BYTES = size_t{1} << 20;

        /**
         * @brief Reads the .npy version and header length, accepting versions 1.0, 2.0 and 3.0.
         * @return The header length in bytes.
         * @throws IOFailure with MalformedFile on any other version or an oversized header, and
         *         with Truncated on a short file.
         */
        size_t read_npy_header_length(std::ifstream& ifs, const std::string& path)
        {
            uint8_t version[2];
            read_exact(ifs, reinterpret_cast<char*>(version), 2, "format version", path);

            const bool known = version[1] == 0 &&
                               (version[0] == 1 || version[0] == 2 || version[0] == 3);
            if (!known)
                fail_io(IOStatus::MalformedFile,
                        "File {} has .npy format version {}.{}; versions 1.0, 2.0 and 3.0 load.",
                        path, static_cast<int>(version[0]), static_cast<int>(version[1]));

            // Version 1.0 stores the length in two little-endian bytes, 2.0 and 3.0 in four.
            const size_t field_bytes = version[0] == 1 ? 2 : 4;
            uint8_t field[4] = {0, 0, 0, 0};
            read_exact(ifs, reinterpret_cast<char*>(field), field_bytes, "header length", path);

            size_t length = 0;
            for (size_t i = field_bytes; i > 0; --i)
                length = (length << 8) | field[i - 1];

            if (length > K_MAX_NPY_HEADER_BYTES)
                fail_io(IOStatus::MalformedFile,
                        "File {} declares a .npy header of {} bytes; at most {} are read.", path,
                        length, K_MAX_NPY_HEADER_BYTES);
            return length;
        }

        /**
         * @brief Throws unless the host buffer size equals the tensor's data size.
         * @throws std::invalid_argument naming op on a mismatch.
         */
        void require_payload_bytes(const Tensor& t, size_t bytes, const char* op)
        {
            const size_t expected = payload_bytes(t);
            if (bytes != expected)
                Dispatch::Detail::throw_operand_error(
                    op, "t", "holds {} bytes of data, but the host buffer has {}", expected, bytes);
        }
    } // namespace

    /** @brief Internal header for .sushi files */
    struct SushiHeader
    {
        char magic[6] = "SUSHI";
        uint32_t version = K_SUSHI_FORMAT_VERSION;
        int32_t rank;
        int64_t shape[Core::K_MAX_TENSOR_RANK];
        int32_t dtype;
        int32_t layout;
    };

    IOStatus IO::save(const Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.save");

        // Padding bytes and unused shape slots are written as zero.
        SushiHeader header;
        std::memset(&header, 0, sizeof(SushiHeader));
        std::memcpy(header.magic, "SUSHI", sizeof(header.magic));
        header.version = K_SUSHI_FORMAT_VERSION;
        header.rank = t.rank;
        header.dtype = static_cast<int32_t>(t.dtype);
        header.layout = static_cast<int32_t>(t.layout);

        for (int i = 0; i < t.rank; ++i)
            header.shape[i] = t.shape[i];

        const IOStatus ran = run_before_file(engine_);
        if (ran != IOStatus::Ok)
            return ran;

        const IOStatus status = status_of([&]()
        {
            write_file(path, [&](std::ofstream& ofs)
            {
                write_checked(ofs, &header, sizeof(SushiHeader), "header", path);
                write_payload(ofs, t, path);
            });
        });

        if (status == IOStatus::Ok)
            SB_LOG_INFO(K_IO_LOG_CATEGORY, "Saved native SushiBLAS file: {}", path);
        return status;
    }

    IOStatus IO::load(Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.load");

        const IOStatus ran = run_before_file(engine_);
        if (ran != IOStatus::Ok)
            return ran;

        return status_of([&]()
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.is_open())
                fail_io(IOStatus::FileUnreadable, "Failed to open file for reading: {}", path);

            SushiHeader header;
            read_exact(ifs, reinterpret_cast<char*>(&header), sizeof(SushiHeader), "header", path);

            if (std::string(header.magic, 5) != "SUSHI")
                fail_io(IOStatus::MalformedFile, "File {} does not start with the .sushi magic.",
                        path);
            if (header.version != K_SUSHI_FORMAT_VERSION)
                fail_io(IOStatus::MalformedFile,
                        "File {} has .sushi format version {}; this build reads version {} only.",
                        path, header.version, K_SUSHI_FORMAT_VERSION);
            if (header.rank != t.rank)
                fail_io(IOStatus::ShapeMismatch,
                        "File {} holds rank {}; the tensor it is loaded into has rank {}.", path,
                        header.rank, t.rank);

            for (int i = 0; i < t.rank; ++i)
                if (header.shape[i] != t.shape[i])
                    fail_io(IOStatus::ShapeMismatch,
                            "File {} holds {} in dimension {}; the tensor it is loaded into "
                            "has {}.",
                            path, header.shape[i], i, t.shape[i]);

            if (header.dtype != static_cast<int32_t>(t.dtype))
                fail_io(IOStatus::DtypeMismatch,
                        "File {} holds dtype id {}; the tensor it is loaded into is {}.", path,
                        header.dtype, Core::to_string(t.dtype));
            if (header.layout != static_cast<int32_t>(t.layout))
                fail_io(IOStatus::LayoutMismatch,
                        "File {} holds layout id {}; the tensor it is loaded into has layout "
                        "id {}.",
                        path, header.layout, static_cast<int32_t>(t.layout));

            read_payload(ifs, t, path);
        });
    }

    IOStatus IO::save_bin(const Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.save_bin");

        const IOStatus ran = run_before_file(engine_);
        if (ran != IOStatus::Ok)
            return ran;

        return status_of([&]()
        {
            write_file(path, [&](std::ofstream& ofs)
            {
                write_payload(ofs, t, path);
            });
        });
    }

    IOStatus IO::load_bin(Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.load_bin");

        const IOStatus ran = run_before_file(engine_);
        if (ran != IOStatus::Ok)
            return ran;

        return status_of([&]()
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.is_open())
                fail_io(IOStatus::FileUnreadable, "Failed to open file for reading: {}", path);

            read_payload(ifs, t, path);
        });
    }

    IOStatus IO::save_npy(const Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.save_npy");
        require_npy_layout(t, "io.save_npy");

        const IOStatus ran = run_before_file(engine_);
        if (ran != IOStatus::Ok)
            return ran;

        const char magic[] = "\x93NUMPY";
        uint8_t major = 1;
        uint8_t minor = 0;

        std::string dict = "{'descr': '";
        // A dtype with no .npy descriptor throws; see modules/operations/README.md.
        dict += npy_descr(t.dtype);

        dict += "', 'fortran_order': ";
        dict += (t.layout == Core::Layout::ColumnMajor) ? "True" : "False";
        dict += ", 'shape': (";

        for (int i = 0; i < t.rank; ++i)
        {
            dict += std::to_string(t.shape[i]);

            if (t.rank == 1 || i < t.rank - 1)
                dict += ",";
            if (i < t.rank - 1)
                dict += " ";
        }
        dict += "), }";

        while ((10 + dict.length() + 1) % 64 != 0)
            dict += " ";

        dict += "\n";

        uint16_t header_len = static_cast<uint16_t>(dict.length());

        const IOStatus status = status_of([&]()
        {
            write_file(path, [&](std::ofstream& ofs)
            {
                write_checked(ofs, magic, 6, "magic string", path);
                write_checked(ofs, &major, 1, "format version", path);
                write_checked(ofs, &minor, 1, "format version", path);
                write_checked(ofs, &header_len, 2, "header length", path);
                write_checked(ofs, dict.c_str(), header_len, "header", path);

                write_payload(ofs, t, path);
            });
        });

        if (status == IOStatus::Ok)
            SB_LOG_INFO(K_IO_LOG_CATEGORY, "Exported NumPy file: {}", path);
        return status;
    }

    IOStatus IO::load_npy(Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.load_npy");
        require_npy_layout(t, "io.load_npy");

        const IOStatus ran = run_before_file(engine_);
        if (ran != IOStatus::Ok)
            return ran;

        const IOStatus status = status_of([&]()
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.is_open())
                fail_io(IOStatus::FileUnreadable, "Failed to open file for reading: {}", path);

            char magic[6];
            read_exact(ifs, magic, 6, "magic string", path);
            if (std::string(magic, 6) != "\x93NUMPY")
                fail_io(IOStatus::MalformedFile, "File {} does not start with the .npy magic.",
                        path);

            const size_t header_len = read_npy_header_length(ifs, path);

            std::string text(header_len, ' ');
            if (header_len > 0)
                read_exact(ifs, &text[0], header_len, "header", path);

            const NpyHeader header = NpyHeaderParser(text, path).parse();
            require_npy_header_matches(header, t, path);

            read_payload(ifs, t, path);
        });

        if (status == IOStatus::Ok)
            SB_LOG_INFO(K_IO_LOG_CATEGORY, "Loaded NumPy file: {}", path);
        return status;
    }

    void IO::read(const Tensor& t, void* dst, size_t bytes)
    {
        require_dense_payload(t, "io.read");
        require_payload_bytes(t, bytes, "io.read");

        run_recorded(engine_);

        if (bytes > 0)
            storage_queue(t).memcpy(dst, t.data(), bytes).wait();
    }

    void IO::write(Tensor& t, const void* src, size_t bytes)
    {
        require_dense_payload(t, "io.write");
        require_payload_bytes(t, bytes, "io.write");

        run_recorded(engine_);

        if (bytes > 0)
            storage_queue(t).memcpy(t.data(), src, bytes).wait();
    }

    void IO::require_element_type(const Tensor& t, Core::DataType expected, const char* op)
    {
        Dispatch::require_dtype(t, expected, op, "t");
    }

    std::string IO::to_string(const Tensor& t, int precision, int edge_items)
    {
        run_recorded(engine_);

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

        ss << "):" << std::endl;
        ss << std::fixed << std::setprecision(precision);

        ElementRunReader reader(t);
        const TextLayout layout{t.shape.data(), t.strides.data(), t.rank, precision, edge_items};
        write_dimension(ss, reader, layout, 0, 0);

        return ss.str();
    }
} // namespace SushiBLAS
