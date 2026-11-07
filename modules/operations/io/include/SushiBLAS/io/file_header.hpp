/**************************************************************************/
/* file_header.hpp                                                        */
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
 * @file file_header.hpp
 * @brief Declares the readers and checks of the .npy and .sushi file headers, which work on bytes and text and touch no file.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/io/io_status.hpp>

namespace SushiBLAS
{
    /** @brief Carries why a header was refused: the status the IO call returns and the message it logs. */
    struct HeaderRefusal
    {
        IOStatus status;     /**< Why the header was refused. */
        std::string message; /**< Names the file and the defect. */
    };

    /** @brief Holds the version the .sushi header carries and the only one this build reads. */
    inline constexpr uint32_t K_SUSHI_FORMAT_VERSION = 1;

    /** @brief Holds the most bytes a .npy prefix occupies: six of magic, two of version, four of length. */
    inline constexpr std::size_t K_NPY_PREFIX_MAX_BYTES = 12;

    /** @brief Holds the largest .npy header dictionary this build reads, in bytes. */
    inline constexpr std::size_t K_MAX_NPY_HEADER_BYTES = std::size_t{1} << 20;

    /** @brief Holds the fields of a .npy header dictionary. */
    struct NpyHeader
    {
        std::string descr;          /**< The dtype descriptor, such as "<f4". */
        bool fortran_order = false; /**< Whether the payload is in Fortran order. */
        std::vector<int64_t> shape; /**< The dimensions, outermost first. */
    };

    /** @brief Holds what the first bytes of a .npy file say about the rest of its header. */
    struct NpyPrefix
    {
        std::size_t header_bytes; /**< The length of the header dictionary that follows. */
        std::size_t prefix_bytes; /**< The bytes the magic, the version and the length occupy. */
    };

    /** @brief Holds what a tensor expects of the .npy file loaded into it. */
    struct NpyExpectation
    {
        std::string descr;          /**< The dtype descriptor the tensor's dtype has. */
        bool column_major = false;  /**< Whether the tensor's layout is column-major. */
        std::vector<int64_t> shape; /**< The tensor's dimensions; its rank is the size. */
    };

    /** @brief Is the fixed-size header of a .sushi file, in the byte order the file stores. */
    struct SushiHeader
    {
        char magic[6] = "SUSHI";
        uint32_t version = K_SUSHI_FORMAT_VERSION;
        int32_t rank;
        int64_t shape[Core::K_MAX_TENSOR_RANK];
        int32_t dtype;
        int32_t layout;
    };

    /** @brief Holds what a tensor expects of the .sushi file loaded into it. */
    struct SushiExpectation
    {
        int32_t rank = 0;           /**< The tensor's rank. */
        std::vector<int64_t> shape; /**< The tensor's dimensions. */
        int32_t dtype = 0;          /**< The tensor's dtype id. */
        const char* dtype_name = ""; /**< The tensor's dtype name, for the message. */
        int32_t layout = 0;         /**< The tensor's layout id. */
    };

    /**
     * @brief Reads the magic, the version and the header length of a .npy file from its first bytes.
     * @param bytes The first bytes of the file; @p size of them are valid, at most K_NPY_PREFIX_MAX_BYTES.
     * @return The header length and the prefix size; versions 1.0, 2.0 and 3.0 load.
     * @throws HeaderRefusal with MalformedFile on a wrong magic, version or oversized header, and
     *         with Truncated when the bytes end inside the prefix.
     */
    [[nodiscard]] NpyPrefix parse_npy_prefix(const unsigned char* bytes, std::size_t size,
                                             const std::string& path);

    /**
     * @brief Reads the header dictionary of a .npy file, a Python literal.
     * @throws HeaderRefusal with MalformedFile on malformed text or a wrong set of keys.
     */
    [[nodiscard]] NpyHeader parse_npy_header(const std::string& text, const std::string& path);

    /**
     * @brief Checks that a parsed .npy header gives exactly the dtype, order and shape expected.
     * @throws HeaderRefusal with the mismatch of the first field that differs.
     */
    void require_npy_header_matches(const NpyHeader& header, const NpyExpectation& expected,
                                    const std::string& path);

    /**
     * @brief Checks the magic, the version, the rank, the shape, the dtype and the layout of a .sushi header.
     * @throws HeaderRefusal with MalformedFile for the magic or the version, and with the mismatch
     *         of the first field that differs otherwise.
     */
    void require_sushi_header_matches(const SushiHeader& header, const SushiExpectation& expected,
                                      const std::string& path);
} // namespace SushiBLAS
