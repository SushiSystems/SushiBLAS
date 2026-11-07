/**************************************************************************/
/* file_remover.hpp                                                       */
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
 * @file file_remover.hpp
 * @brief Declares FileRemover, the port through which a failed save deletes its partial file, and DiskFileRemover.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#pragma once

#include <cstdio>
#include <string>

namespace SushiBLAS
{
    namespace IOFiles
    {
        /** @brief Deletes a file by path and says whether the file is gone. */
        class FileRemover
        {
            public:
                FileRemover() = default;
                FileRemover(const FileRemover&) = delete;
                FileRemover& operator=(const FileRemover&) = delete;
                FileRemover(FileRemover&&) = delete;
                FileRemover& operator=(FileRemover&&) = delete;
                virtual ~FileRemover() = default;

                /**
                 * @brief Deletes the file at @p path.
                 * @return True when the file is gone; false when the delete failed.
                 */
                [[nodiscard]] virtual bool remove(const std::string& path) const = 0;
        };

        /** @brief Deletes a file from the disk. */
        class DiskFileRemover final : public FileRemover
        {
            public:
                /** @copydoc FileRemover::remove */
                [[nodiscard]] bool remove(const std::string& path) const override
                {
                    return std::remove(path.c_str()) == 0;
                }
        };
    } // namespace IOFiles
} // namespace SushiBLAS
