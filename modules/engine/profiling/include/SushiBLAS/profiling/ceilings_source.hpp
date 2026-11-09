/**************************************************************************/
/* ceilings_source.hpp                                                    */
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
 * @file ceilings_source.hpp
 * @brief Declares CeilingsSource, the port a BlasEnvironment reads the text of a ceilings file through.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#pragma once

#include <string>
#include <optional>

namespace SushiBLAS
{
    namespace Profiling
    {
        /**
         * @brief Hands out the text of a ceilings file by its name.
         *
         * FolderCeilingsSource reads a folder on disk and InMemoryCeilingsSource holds the texts,
         * so a host that keeps its ceilings elsewhere adds one more adapter and a test needs no folder.
         */
        class CeilingsSource
        {
            public:
                virtual ~CeilingsSource() = default;

                /**
                 * @brief Returns the text of a ceilings file.
                 * @param file_name The name `ceilings_file_name` gives a device.
                 * @return The text, or nothing when the source holds no such file.
                 */
                [[nodiscard]] virtual std::optional<std::string> read(const std::string& file_name) const = 0;

                /** @brief Returns where a file of that name is looked for, in words a log record can carry. */
                [[nodiscard]] virtual std::string location(const std::string& file_name) const = 0;
        };
    } // namespace Profiling
} // namespace SushiBLAS
