/**************************************************************************/
/* folder_ceilings_source.cpp                                             */
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
 * @file folder_ceilings_source.cpp
 * @brief Defines FolderCeilingsSource.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#include <SushiBLAS/profiling/folder_ceilings_source.hpp>

#include <utility>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace SushiBLAS
{
    namespace Profiling
    {
        FolderCeilingsSource::FolderCeilingsSource(std::string folder)
            : folder_(std::move(folder))
        {

        }

        std::optional<std::string> FolderCeilingsSource::read(const std::string& file_name) const
        {
            std::ifstream file(std::filesystem::path(folder_) / file_name, std::ios::binary);
            if (!file)
                return std::nullopt;
            std::ostringstream contents;
            contents << file.rdbuf();
            return contents.str();
        }

        std::string FolderCeilingsSource::location(const std::string& file_name) const
        {
            return (std::filesystem::path(folder_) / file_name).string();
        }
    } // namespace Profiling
} // namespace SushiBLAS
