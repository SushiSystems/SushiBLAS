/**************************************************************************/
/* file_header.cpp                                                        */
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
 * @file file_header.cpp
 * @brief Defines the readers and checks of the .npy and .sushi file headers.
 * @author Mustafa Garip
 */

#include <SushiBLAS/io/file_header.hpp>

#include <limits>
#include <cstring>
#include <string_view>
#include <SushiLog/text/text_format.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Throws a HeaderRefusal whose message is the formatted text. */
        template<typename... Args>
        [[noreturn]] void refuse(IOStatus status, std::string_view format, const Args&... args)
        {
            throw HeaderRefusal{status, SushiLog::format_text(format, args...)};
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
                 * @throws HeaderRefusal with MalformedFile on malformed text or a wrong set of keys.
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
                    refuse(IOStatus::MalformedFile,
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

    } // namespace

    NpyPrefix parse_npy_prefix(const unsigned char* bytes, std::size_t size, const std::string& path)
    {
        constexpr std::size_t magic_bytes = 6;
        constexpr std::size_t version_end = 8;

        if (size < magic_bytes)
            refuse(IOStatus::Truncated,
                   "File {} ends inside its magic string: {} bytes expected, {} read; nothing was "
                   "loaded.", path, magic_bytes, size);
        if (std::memcmp(bytes, "\x93NUMPY", magic_bytes) != 0)
            refuse(IOStatus::MalformedFile, "File {} does not start with the .npy magic.", path);
        if (size < version_end)
            refuse(IOStatus::Truncated,
                   "File {} ends inside its format version: {} bytes expected, {} read; nothing "
                   "was loaded.", path, version_end - magic_bytes, size - magic_bytes);

        const unsigned char* version = bytes + magic_bytes;
        const bool known = version[1] == 0 && (version[0] == 1 || version[0] == 2 || version[0] == 3);
        if (!known)
            refuse(IOStatus::MalformedFile,
                   "File {} has .npy format version {}.{}; versions 1.0, 2.0 and 3.0 load.", path,
                   static_cast<int>(version[0]), static_cast<int>(version[1]));

        // Version 1.0 stores the length in two little-endian bytes, 2.0 and 3.0 in four.
        const std::size_t field_bytes = version[0] == 1 ? 2 : 4;
        if (size < version_end + field_bytes)
            refuse(IOStatus::Truncated,
                   "File {} ends inside its header length: {} bytes expected, {} read; nothing "
                   "was loaded.", path, field_bytes, size - version_end);

        std::size_t length = 0;
        for (std::size_t i = field_bytes; i > 0; --i)
            length = (length << 8) | bytes[version_end + i - 1];

        if (length > K_MAX_NPY_HEADER_BYTES)
            refuse(IOStatus::MalformedFile,
                   "File {} declares a .npy header of {} bytes; at most {} are read.", path,
                   length, K_MAX_NPY_HEADER_BYTES);
        return NpyPrefix{length, version_end + field_bytes};
    }

    NpyHeader parse_npy_header(const std::string& text, const std::string& path)
    {
        return NpyHeaderParser(text, path).parse();
    }

    void require_npy_header_matches(const NpyHeader& header, const NpyExpectation& expected,
                                    const std::string& path)
    {
        if (header.descr != expected.descr)
            refuse(IOStatus::DtypeMismatch,
                   "File {} holds .npy dtype '{}'; the tensor it is loaded into needs '{}'.",
                   path, header.descr, expected.descr);

        // At rank 0 and 1 C and Fortran order lay the elements out identically.
        if (expected.shape.size() >= 2 && header.fortran_order != expected.column_major)
            refuse(IOStatus::LayoutMismatch,
                   "File {} has fortran_order {}; the tensor it is loaded into is {}.", path,
                   header.fortran_order ? "True" : "False",
                   expected.column_major ? "ColumnMajor" : "RowMajor");

        if (header.shape != expected.shape)
            refuse(IOStatus::ShapeMismatch,
                   "File {} holds a .npy array of rank {} whose shape differs from the "
                   "tensor's (rank {}).", path, header.shape.size(), expected.shape.size());
    }

    void require_sushi_header_matches(const SushiHeader& header, const SushiExpectation& expected,
                                      const std::string& path)
    {
        if (std::string(header.magic, 5) != "SUSHI")
            refuse(IOStatus::MalformedFile, "File {} does not start with the .sushi magic.", path);
        if (header.version != K_SUSHI_FORMAT_VERSION)
            refuse(IOStatus::MalformedFile,
                   "File {} has .sushi format version {}; this build reads version {} only.",
                   path, header.version, K_SUSHI_FORMAT_VERSION);
        if (header.rank != expected.rank)
            refuse(IOStatus::ShapeMismatch,
                   "File {} holds rank {}; the tensor it is loaded into has rank {}.", path,
                   header.rank, expected.rank);

        for (int i = 0; i < expected.rank; ++i)
            if (header.shape[i] != expected.shape[static_cast<std::size_t>(i)])
                refuse(IOStatus::ShapeMismatch,
                       "File {} holds {} in dimension {}; the tensor it is loaded into has {}.",
                       path, header.shape[i], i, expected.shape[static_cast<std::size_t>(i)]);

        if (header.dtype != expected.dtype)
            refuse(IOStatus::DtypeMismatch,
                   "File {} holds dtype id {}; the tensor it is loaded into is {}.", path,
                   header.dtype, expected.dtype_name);
        if (header.layout != expected.layout)
            refuse(IOStatus::LayoutMismatch,
                   "File {} holds layout id {}; the tensor it is loaded into has layout id {}.",
                   path, header.layout, expected.layout);
    }
} // namespace SushiBLAS
