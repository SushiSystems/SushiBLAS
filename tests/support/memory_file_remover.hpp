/**************************************************************************/
/* memory_file_remover.hpp                                                */
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
 * @file memory_file_remover.hpp
 * @brief Declares MemoryFileRemover, a FileRemover over a set of paths held in memory.
 * @author Mustafa Garip
 */

#pragma once

#include <set>
#include <string>
#include <SushiBLAS/io/file_remover.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a set of paths as the files it can delete; touches no disk. */
        class MemoryFileRemover final : public IOFiles::FileRemover
        {
            public:
                /** @brief Adds a file at @p path. */
                void add(const std::string& path) { files.insert(path); }

                /** @brief True when a file is held at @p path. */
                [[nodiscard]] bool has(const std::string& path) const { return files.count(path) != 0; }

                /** @copydoc IOFiles::FileRemover::remove */
                [[nodiscard]] bool remove(const std::string& path) const override
                {
                    return files.erase(path) != 0;
                }

            private:
                /// @brief The paths still held; erased through a const remove, as a disk delete is.
                mutable std::set<std::string> files;
        };
    } // namespace Test
} // namespace SushiBLAS
