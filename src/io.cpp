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

#include <limits>
#include <vector>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <SushiBLAS/io.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Holds the only .sushi format version this build reads and writes. */
        constexpr uint32_t K_SUSHI_FORMAT_VERSION = 1;

        /** @brief Returns the number of bytes the tensor's elements occupy. */
        size_t payload_bytes(const Tensor& t)
        {
            return static_cast<size_t>(t.num_elements) * Core::bytes_per_element(t.dtype);
        }

        /** @brief Reports whether the host reaches the tensor's storage only through the queue. */
        bool is_device_allocated(const Tensor& t)
        {
            return t.storage->get_residency() == SushiRuntime::API::Residency::Device;
        }

        /**
         * @brief Returns the queue of the device that holds the tensor's storage.
         * @pre t has storage.
         */
        sycl::queue& storage_queue(const Tensor& t)
        {
            return t.storage->get_context().get_queue(t.storage->get_device_index());
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
                case Core::DataType::FLOAT32:   return "<f4";
                case Core::DataType::FLOAT64:   return "<f8";
                case Core::DataType::COMPLEX32: return "<c8";
                case Core::DataType::COMPLEX64: return "<c16";
                case Core::DataType::HALF:      return "<f2";
                case Core::DataType::INT32:     return "<i4";
                case Core::DataType::INT64:     return "<i8";
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
                 * @throws std::runtime_error on malformed text, an unknown key or a missing key.
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
                /** @brief Throws a runtime_error naming the file and the defect. */
                [[noreturn]] void fail(const std::string& what) const
                {
                    SB_THROW_IF(true, "The .npy header of {} is malformed at byte {}: {}.", path_,
                                pos_, what);
                    throw std::runtime_error(what);
                }

                /** @brief Marks a key as seen and throws when it was seen before. */
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

                /** @brief Skips space and consumes @p c; throws when another character is there. */
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
         * @brief Throws unless a parsed .npy header gives exactly the tensor's dtype, order, shape.
         * @throws std::runtime_error naming the first field that differs.
         */
        void require_npy_header_matches(const NpyHeader& header, const Tensor& t,
                                        const std::string& path)
        {
            const std::string expected_descr = npy_descr(t.dtype);
            SB_THROW_IF(header.descr != expected_descr,
                        "File {} holds .npy dtype '{}'; the tensor it is loaded into needs '{}'.",
                        path, header.descr, expected_descr);

            const bool column_major = t.layout == Core::Layout::COLUMN_MAJOR;

            // At rank 0 and 1 C and Fortran order lay the elements out identically.
            SB_THROW_IF(t.rank >= 2 && header.fortran_order != column_major,
                        "File {} has fortran_order {}; the tensor it is loaded into is {}.", path,
                        header.fortran_order ? "True" : "False",
                        column_major ? "COLUMN_MAJOR" : "ROW_MAJOR");

            bool same_shape = header.shape.size() == static_cast<size_t>(t.rank);
            for (int i = 0; same_shape && i < t.rank; ++i)
                same_shape = header.shape[i] == t.shape[i];
            SB_THROW_IF(!same_shape,
                        "File {} holds a .npy array of rank {} whose shape differs from the "
                        "tensor's (rank {}).", path, header.shape.size(), t.rank);
        }

        /** @brief Reads exactly @p bytes into @p destination; throws when the file ends first. */
        void read_exact(std::ifstream& ifs, char* destination, size_t bytes,
                        const char* part, const std::string& path)
        {
            ifs.read(destination, static_cast<std::streamsize>(bytes));

            const size_t bytes_read = static_cast<size_t>(ifs.gcount());
            SB_THROW_IF(bytes_read != bytes,
                        "File {} ends inside its {}: {} bytes expected, {} read; nothing was "
                        "loaded.",
                        path, part, bytes, bytes_read);
        }

        /** @brief Writes @p bytes from @p source and throws when the stream reports a failure. */
        void write_checked(std::ofstream& ofs, const void* source, size_t bytes, const char* part,
                           const std::string& path)
        {
            ofs.write(static_cast<const char*>(source), static_cast<std::streamsize>(bytes));
            SB_THROW_IF(!ofs, "Writing the {} of {} failed after {} bytes were requested.",
                        part, path, bytes);
        }

        /** @brief Writes the tensor's bytes to the stream, copying device storage to host first. */
        void write_payload(std::ofstream& ofs, const Tensor& t, const std::string& path)
        {
            const size_t bytes = payload_bytes(t);
            if (bytes == 0)
                return;

            if (is_device_allocated(t))
            {
                std::vector<char> buf(bytes);
                storage_queue(t).memcpy(buf.data(), t.data(), bytes).wait();
                write_checked(ofs, buf.data(), bytes, "tensor data", path);
            }
            else
                write_checked(ofs, t.data(), bytes, "tensor data", path);
        }

        /**
         * @brief Creates the file at @p path, runs @p write_body on it, deletes it on failure.
         * @throws std::runtime_error when the file cannot be opened or a write or the close fails.
         */
        template<typename WriteBody>
        void write_file(const std::string& path, WriteBody write_body)
        {
            std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
            SB_THROW_IF(!ofs.is_open(), "Failed to open file for writing: {}", path);

            try
            {
                write_body(ofs);
                ofs.close();
                SB_THROW_IF(ofs.fail(), "Closing {} failed; the file may be incomplete.", path);
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
            if (t.layout == Core::Layout::COLUMN_MAJOR && t.rank >= 3)
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
                case Core::DataType::HALF:
                    return {static_cast<double>(static_cast<float>(
                                reinterpret_cast<const sycl::half*>(buf)[index])), 0.0};
                case Core::DataType::FLOAT32:
                    return {reinterpret_cast<const float*>(buf)[index], 0.0};
                case Core::DataType::FLOAT64:
                    return {reinterpret_cast<const double*>(buf)[index], 0.0};
                case Core::DataType::COMPLEX32:
                {
                    const float* c = reinterpret_cast<const float*>(buf) + index * 2;
                    return {c[0], c[1]};
                }
                case Core::DataType::COMPLEX64:
                {
                    const double* c = reinterpret_cast<const double*>(buf) + index * 2;
                    return {c[0], c[1]};
                }
                case Core::DataType::INT32:
                    return {static_cast<double>(reinterpret_cast<const int32_t*>(buf)[index]), 0.0};
                case Core::DataType::INT64:
                    return {static_cast<double>(reinterpret_cast<const int64_t*>(buf)[index]), 0.0};
                default:
                    SB_THROW_IF(true, "Cannot render a tensor of dtype {} as text.",
                                Core::to_string(dtype));
                    return {0.0, 0.0};
            }
        }

        /** @brief Reads the tensor's bytes from the stream and stores them once all arrived. */
        void read_payload(std::ifstream& ifs, Tensor& t, const std::string& path)
        {
            const size_t bytes = payload_bytes(t);
            if (bytes == 0)
                return;

            std::vector<char> buf(bytes);
            read_exact(ifs, buf.data(), bytes, "tensor data", path);

            if (is_device_allocated(t))
                storage_queue(t).memcpy(t.data(), buf.data(), bytes).wait();
            else
                std::memcpy(t.data(), buf.data(), bytes);
        }

        /** @brief Holds the largest .npy header dictionary this build reads, in bytes. */
        constexpr size_t K_MAX_NPY_HEADER_BYTES = size_t{1} << 20;

        /**
         * @brief Reads the .npy version and header length, accepting versions 1.0, 2.0 and 3.0.
         * @return The header length in bytes.
         * @throws std::runtime_error on any other version, an oversized header or a short file.
         */
        size_t read_npy_header_length(std::ifstream& ifs, const std::string& path)
        {
            uint8_t version[2];
            read_exact(ifs, reinterpret_cast<char*>(version), 2, "format version", path);

            const bool known = version[1] == 0 &&
                               (version[0] == 1 || version[0] == 2 || version[0] == 3);
            SB_THROW_IF(!known,
                        "File {} has .npy format version {}.{}; versions 1.0, 2.0 and 3.0 load.",
                        path, static_cast<int>(version[0]), static_cast<int>(version[1]));

            // Version 1.0 stores the length in two little-endian bytes, 2.0 and 3.0 in four.
            const size_t field_bytes = version[0] == 1 ? 2 : 4;
            uint8_t field[4] = {0, 0, 0, 0};
            read_exact(ifs, reinterpret_cast<char*>(field), field_bytes, "header length", path);

            size_t length = 0;
            for (size_t i = field_bytes; i > 0; --i)
                length = (length << 8) | field[i - 1];

            SB_THROW_IF(length > K_MAX_NPY_HEADER_BYTES,
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
        int64_t shape[Core::MAX_TENSOR_RANK];
        int32_t dtype;
        int32_t layout;
    };

    void IO::save(const Tensor& t, const std::string& path)
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

        engine_.execute();

        write_file(path, [&](std::ofstream& ofs)
        {
            write_checked(ofs, &header, sizeof(SushiHeader), "header", path);
            write_payload(ofs, t, path);
        });

        SB_LOG_INFO("Saved native SushiBLAS file: {}", path);
    }

    void IO::load(Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.load");

        engine_.execute();

        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file for reading: {}", path);

        SushiHeader header;
        read_exact(ifs, reinterpret_cast<char*>(&header), sizeof(SushiHeader), "header", path);

        SB_THROW_IF(std::string(header.magic, 5) != "SUSHI", "Invalid .sushi file magic.");
        SB_THROW_IF(header.version != K_SUSHI_FORMAT_VERSION,
                    "File {} has .sushi format version {}; this build reads version {} only.",
                    path, header.version, K_SUSHI_FORMAT_VERSION);
        SB_THROW_IF(header.rank != t.rank, "Rank mismatch in .sushi file.");

        for (int i = 0; i < t.rank; ++i)
            SB_THROW_IF(header.shape[i] != t.shape[i], "Dimension mismatch at index {}", i);

        SB_THROW_IF(header.dtype != static_cast<int32_t>(t.dtype),
                    "File {} holds dtype id {}; the tensor it is loaded into is {}.",
                    path, header.dtype, Core::to_string(t.dtype));
        SB_THROW_IF(header.layout != static_cast<int32_t>(t.layout),
                    "File {} holds layout id {}; the tensor it is loaded into has layout id {}.",
                    path, header.layout, static_cast<int32_t>(t.layout));

        read_payload(ifs, t, path);
    }

    void IO::save_bin(const Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.save_bin");

        engine_.execute();

        write_file(path, [&](std::ofstream& ofs)
        {
            write_payload(ofs, t, path);
        });
    }

    void IO::load_bin(Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.load_bin");

        engine_.execute();

        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file: {}", path);

        read_payload(ifs, t, path);
    }

    void IO::save_npy(const Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.save_npy");
        require_npy_layout(t, "io.save_npy");

        engine_.execute();

        const char magic[] = "\x93NUMPY";
        uint8_t major = 1;
        uint8_t minor = 0;

        std::string dict = "{'descr': '";
        // A dtype with no .npy descriptor throws; see src/README.md.
        dict += npy_descr(t.dtype);

        dict += "', 'fortran_order': ";
        dict += (t.layout == Core::Layout::COLUMN_MAJOR) ? "True" : "False";
        dict += ", 'shape': (";

        for (int i = 0; i < t.rank; ++i)
        {
            dict += std::to_string(t.shape[i]);

            if (t.rank == 1 || i < t.rank - 1) dict += ",";
            if (i < t.rank - 1) dict += " ";
        }
        dict += "), }";

        while ((10 + dict.length() + 1) % 64 != 0) dict += " ";

        dict += "\n";

        uint16_t header_len = static_cast<uint16_t>(dict.length());

        write_file(path, [&](std::ofstream& ofs)
        {
            write_checked(ofs, magic, 6, "magic string", path);
            write_checked(ofs, &major, 1, "format version", path);
            write_checked(ofs, &minor, 1, "format version", path);
            write_checked(ofs, &header_len, 2, "header length", path);
            write_checked(ofs, dict.c_str(), header_len, "header", path);

            write_payload(ofs, t, path);
        });

        SB_LOG_INFO("Exported NumPy file: {}", path);
    }

    void IO::load_npy(Tensor& t, const std::string& path)
    {
        require_dense_payload(t, "io.load_npy");
        require_npy_layout(t, "io.load_npy");

        engine_.execute();

        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file for reading: {}", path);

        char magic[6];
        read_exact(ifs, magic, 6, "magic string", path);
        SB_THROW_IF(std::string(magic, 6) != "\x93NUMPY", "Invalid .npy file magic.");

        const size_t header_len = read_npy_header_length(ifs, path);

        std::string text(header_len, ' ');
        if (header_len > 0)
            read_exact(ifs, &text[0], header_len, "header", path);

        const NpyHeader header = NpyHeaderParser(text, path).parse();
        require_npy_header_matches(header, t, path);

        read_payload(ifs, t, path);

        SB_LOG_INFO("Loaded NumPy file: {}", path);
    }

    void IO::read(const Tensor& t, void* dst, size_t bytes)
    {
        require_dense_payload(t, "io.read");
        require_payload_bytes(t, bytes, "io.read");

        engine_.execute();

        if (bytes > 0)
            storage_queue(t).memcpy(dst, t.data(), bytes).wait();
    }

    void IO::write(Tensor& t, const void* src, size_t bytes)
    {
        require_dense_payload(t, "io.write");
        require_payload_bytes(t, bytes, "io.write");

        engine_.execute();

        if (bytes > 0)
            storage_queue(t).memcpy(t.data(), src, bytes).wait();
    }

    void IO::require_element_type(const Tensor& t, Core::DataType expected, const char* op)
    {
        Dispatch::require_dtype(t, expected, op, "t");
    }

    std::string IO::to_string(const Tensor& t, int precision, int edge_items)
    {
        engine_.execute();

        Dispatch::require_storage(t, "io.to_string", "t");

        const size_t bpe = Core::bytes_per_element(t.dtype);

        // The view's elements lie in [data(), data() + element_span()), addressed by its strides.
        const int64_t span = t.element_span();

        std::vector<char> host_buf(static_cast<size_t>(span) * bpe);
        if (span > 0)
            storage_queue(t).memcpy(host_buf.data(), t.data(), host_buf.size()).wait();

        std::vector<std::pair<double, double>> host_data(static_cast<size_t>(span));

        // INT64 values above 2^53 lose precision through double; see src/README.md.
        for (int64_t i = 0; i < span; ++i)
            host_data[i] = decode_element(host_buf.data(), t.dtype, i);

        std::stringstream ss;
        ss << "Tensor(shape=[";

        for (int i = 0; i < t.rank; ++i)
            ss << t.shape[i] << (i == t.rank - 1 ? "" : ", ");

        ss << "], dtype=";

        if (t.dtype == Core::DataType::HALF) ss << "f16";
        else if (t.dtype == Core::DataType::FLOAT32) ss << "f32";
        else if (t.dtype == Core::DataType::FLOAT64) ss << "f64";
        else if (t.dtype == Core::DataType::COMPLEX32) ss << "c32";
        else if (t.dtype == Core::DataType::COMPLEX64) ss << "c64";
        else if (t.dtype == Core::DataType::INT32) ss << "i32";
        else if (t.dtype == Core::DataType::INT64) ss << "i64";

        ss << "):" << std::endl;
        ss << std::fixed << std::setprecision(precision);

        print_recursive(ss, host_data.data(), t.shape.data(),
                        t.strides.data(), t.rank, 0, 0, precision, edge_items);

        return ss.str();
    }

    void IO::print(const Tensor& t, int precision, int edge_items)
    {
        std::cout << to_string(t, precision, edge_items);
    }

    void IO::print_recursive(std::ostream& os, const std::pair<double, double>* data,
                             const int64_t* shape, const int64_t* strides, int rank,
                             int current_dim, int64_t offset, int precision, int edge_items)
    {
        auto get_val = [&](int64_t idx) -> std::string
        {
            const double re = data[idx].first;
            const double im = data[idx].second;

            if (std::abs(im) < 1e-9)
            {
                std::stringstream val_ss;
                val_ss << std::fixed << std::setprecision(precision) << re;
                return val_ss.str();
            }

            std::stringstream val_ss;
            val_ss << std::fixed << std::setprecision(precision) << "(" << re
                   << (im >= 0 ? "+" : "") << im << "j)";
            return val_ss.str();
        };

        if (current_dim == rank - 1)
        {
            os << "[";

            for (int64_t i = 0; i < shape[current_dim]; ++i)
            {
                if (shape[current_dim] > edge_items * 2 && i == edge_items)
                {
                    os << "... ";
                    i = shape[current_dim] - edge_items - 1;
                    continue;
                }
                os << get_val(offset + i * strides[current_dim])
                   << (i == shape[current_dim] - 1 ? "" : ", ");
            }

            os << "]";
            return;
        }

        os << "[";

        for (int64_t i = 0; i < shape[current_dim]; ++i)
        {
            if (shape[current_dim] > edge_items * 2 && i == edge_items)
            {
                const std::string indent(current_dim + 1, ' ');
                os << "\n" << indent << "...\n" << indent;
                i = shape[current_dim] - edge_items - 1;
                continue;
            }

            print_recursive(os, data, shape, strides, rank, current_dim + 1,
                            offset + i * strides[current_dim], precision, edge_items);

            if (i < shape[current_dim] - 1)
                os << ",\n" << std::string(current_dim + 1, ' ');
        }

        os << "]";

        if (current_dim == 0)
            os << std::endl;
    }
} // namespace SushiBLAS
