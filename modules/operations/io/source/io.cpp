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
 * @file io.cpp
 * @brief Defines IO, which saves and loads tensors as .sushi, raw binary and .npy files and
 *        copies them to and from host memory.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/io/file_header.hpp>
#include <SushiBLAS/io/io.hpp>
#include <SushiBLAS/io/io_log_category.hpp>
#include <SushiBLAS/io/partial_file.hpp>
#include "io_file_stream.hpp"
#include "io_tensor_access.hpp"

namespace SushiBLAS
{
    namespace
    {
        using Detail::fail_io;
        using Detail::is_device_allocated;
        using Detail::payload_bytes;
        using Detail::read_exact;
        using Detail::read_payload;
        using Detail::refusals_as_failures;
        using Detail::status_of;
        using Detail::storage_queue;
        using Detail::write_checked;
        using Detail::write_file;
        using Detail::write_payload;

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
                case Core::DataType::UInt32:    return "<u4";
                default:
                    SB_THROW_IF(true, "Dtype {} has no .npy descriptor.", Core::to_string(dtype));
                    return "";
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

        return status_of([&]()
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.is_open())
                fail_io(IOStatus::FileUnreadable, "Failed to open file for reading: {}", path);

            SushiHeader header;
            read_exact(ifs, reinterpret_cast<char*>(&header), sizeof(SushiHeader), "header", path);

            SushiExpectation expected;
            expected.rank = t.rank;
            expected.shape.assign(t.shape.begin(), t.shape.begin() + t.rank);
            expected.dtype = static_cast<int32_t>(t.dtype);
            expected.dtype_name = Core::to_string(t.dtype);
            expected.layout = static_cast<int32_t>(t.layout);
            refusals_as_failures([&]() { require_sushi_header_matches(header, expected, path); });

            read_payload(ifs, t, path);
        });
    }

    IOStatus IO::save_bin(const Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.save_bin");

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

        const IOStatus status = status_of([&]()
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.is_open())
                fail_io(IOStatus::FileUnreadable, "Failed to open file for reading: {}", path);

            unsigned char prefix_bytes[K_NPY_PREFIX_MAX_BYTES];
            ifs.read(reinterpret_cast<char*>(prefix_bytes), sizeof(prefix_bytes));
            const size_t prefix_read = static_cast<size_t>(ifs.gcount());
            NpyPrefix prefix{};
            refusals_as_failures(
                [&]() { prefix = parse_npy_prefix(prefix_bytes, prefix_read, path); });
            ifs.clear();
            ifs.seekg(static_cast<std::streamoff>(prefix.prefix_bytes), std::ios::beg);

            std::string text(prefix.header_bytes, ' ');
            if (prefix.header_bytes > 0)
                read_exact(ifs, &text[0], prefix.header_bytes, "header", path);

            NpyExpectation expected;
            expected.descr = npy_descr(t.dtype);
            expected.column_major = t.layout == Core::Layout::ColumnMajor;
            expected.shape.assign(t.shape.begin(), t.shape.begin() + t.rank);
            refusals_as_failures([&]()
            {
                require_npy_header_matches(parse_npy_header(text, path), expected, path);
            });

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

        if (bytes > 0)
            storage_queue(t).memcpy(dst, t.data(), bytes).wait();
    }

    void IO::write(Tensor& t, const void* src, size_t bytes)
    {
        require_dense_payload(t, "io.write");
        require_payload_bytes(t, bytes, "io.write");

        if (bytes > 0)
            storage_queue(t).memcpy(t.data(), src, bytes).wait();
    }

    void IO::require_element_type(const Tensor& t, Core::DataType expected, const char* op)
    {
        Dispatch::require_dtype(t, expected, op, "t");
    }
} // namespace SushiBLAS
