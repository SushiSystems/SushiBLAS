/**************************************************************************/
/* io_file_stream.hpp                                                     */
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
 * @file io_file_stream.hpp
 * @brief Declares the checked file reads and writes the IO calls share.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <string_view>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/io/file_header.hpp>
#include <SushiBLAS/io/io_log_category.hpp>
#include <SushiBLAS/io/io_status.hpp>
#include <SushiBLAS/io/partial_file.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiLog/text/text_format.hpp>

namespace SushiBLAS
{
    namespace Detail
    {
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
            SB_LOG_ERROR(K_IO_LOG_CATEGORY, "io: {}", SushiLog::format_text(format, args...));
            throw IOFailure{status};
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

        /** @brief Runs header work and turns the refusal it throws into an IOFailure. */
        template<typename HeaderWork>
        void refusals_as_failures(HeaderWork header_work)
        {
            try
            {
                header_work();
            }
            catch (const HeaderRefusal& refusal)
            {
                fail_io(refusal.status, "{}", refusal.message);
            }
        }

        /**
         * @brief Reads exactly @p bytes into @p destination; fails Truncated on a short file.
         * @throws IOFailure with Truncated when fewer bytes are left.
         */
        void read_exact(std::ifstream& ifs, char* destination, size_t bytes,
                        const char* part, const std::string& path);

        /**
         * @brief Writes @p bytes from @p source; fails FileUnwritable on a stream failure.
         * @throws IOFailure with FileUnwritable when the stream fails.
         */
        void write_checked(std::ofstream& ofs, const void* source, size_t bytes, const char* part,
                           const std::string& path);

        /**
         * @brief Writes the tensor's bytes to the stream, staging device storage one chunk at a
         *        time.
         */
        void write_payload(std::ofstream& ofs, const Tensor& t, const std::string& path);

        /**
         * @brief Reads the tensor's bytes from the stream, straight into host-readable storage
         *        and one chunk at a time into device storage.
         * @throws IOFailure with Truncated before any element changes if the file is too short.
         */
        void read_payload(std::ifstream& ifs, Tensor& t, const std::string& path);

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
                static_cast<void>(IOFiles::remove_partial_file(path));
                throw;
            }
        }
    } // namespace Detail
} // namespace SushiBLAS
