/**************************************************************************/
/* folder_ceilings_source.hpp                                             */
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
 * @file folder_ceilings_source.hpp
 * @brief Declares FolderCeilingsSource, the CeilingsSource that reads a folder on disk.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#pragma once

#include <string>
#include <optional>
#include <SushiBLAS/profiling/ceilings_source.hpp>

namespace SushiBLAS
{
    namespace Profiling
    {
        /** @brief Reads the `<device>.ceilings.json` files of one folder. */
        class FolderCeilingsSource final : public CeilingsSource
        {
            public:
                /** @param folder The folder that holds the ceilings files. */
                explicit FolderCeilingsSource(std::string folder);

                [[nodiscard]] std::optional<std::string> read(const std::string& file_name) const override;

                [[nodiscard]] std::string location(const std::string& file_name) const override;

            private:
                /// @brief The folder the files are read from.
                std::string folder_;
        };
    } // namespace Profiling
} // namespace SushiBLAS
