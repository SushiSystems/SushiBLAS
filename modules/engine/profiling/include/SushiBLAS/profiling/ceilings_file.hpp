/**************************************************************************/
/* ceilings_file.hpp                                                      */
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
 * @file ceilings_file.hpp
 * @brief Declares the reader and the writer of the ceilings file, format version 1.
 * @author Mustafa Garip
 * @see docs/reference/CEILINGS_FORMAT.md
 */

#pragma once

#include <string>
#include <cstdint>
#include <string_view>
#include <SushiBLAS/profiling/ceilings.hpp>

namespace SushiBLAS
{
    namespace Profiling
    {
        /** @brief Holds the version of the format the writer writes and the reader accepts. */
        inline constexpr std::uint32_t K_CEILINGS_FORMAT_VERSION = 1;

        /** @brief Holds the value of the `format` member of every ceilings file. */
        inline constexpr std::string_view K_CEILINGS_FORMAT_NAME = "sushiblas.ceilings";

        /** @brief Names why a ceilings file was accepted or refused. */
        enum class CeilingsReadStatus : std::uint8_t
        {
            Ok,             /**< The file is a version 1 ceilings file and every value is valid. */
            NotJson,        /**< The text is not a JSON document. */
            WrongFormat,    /**< The document is not a ceilings file. */
            UnknownVersion, /**< The file is a ceilings file of a version this reader does not know. */
            Invalid         /**< A member is missing, unexpected, of the wrong kind or out of range. */
        };

        /** @brief Holds what reading a ceilings file produced. */
        struct CeilingsReadResult
        {
            CeilingsReadStatus status = CeilingsReadStatus::Ok; /**< Whether the file was accepted, and why not. */
            std::string message;                                /**< Where and why a file was refused; empty when accepted. */
            DeviceCeilings ceilings;                            /**< The ceilings, when status is Ok. */

            /** @brief Returns whether the file was accepted. */
            [[nodiscard]] bool ok() const noexcept
            {
                return status == CeilingsReadStatus::Ok;
            }
        };

        /**
         * @brief Reads a ceilings file from its text and validates it.
         * @return The ceilings, or the status and the path of the first refusal.
         * @see docs/reference/CEILINGS_FORMAT.md
         */
        [[nodiscard]] CeilingsReadResult read_ceilings(std::string_view text);

        /**
         * @brief Writes ceilings as the text of a version 1 ceilings file.
         * @pre The ceilings would pass read_ceilings; the writer does not check them.
         */
        [[nodiscard]] std::string write_ceilings(const DeviceCeilings& ceilings);

        /**
         * @brief Returns the name of the ceilings file of a device: its name in lower case with
         *        every run of other characters as one underscore, then `.ceilings.json`.
         */
        [[nodiscard]] std::string ceilings_file_name(std::string_view device_name);
    } // namespace Profiling
} // namespace SushiBLAS
