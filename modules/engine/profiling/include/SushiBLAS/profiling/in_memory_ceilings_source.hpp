/**************************************************************************/
/* in_memory_ceilings_source.hpp                                          */
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
 * @file in_memory_ceilings_source.hpp
 * @brief Declares InMemoryCeilingsSource, the CeilingsSource that holds its texts in memory.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#pragma once

#include <map>
#include <string>
#include <utility>
#include <optional>
#include <SushiBLAS/profiling/ceilings_source.hpp>

namespace SushiBLAS
{
    namespace Profiling
    {
        /** @brief Holds ceilings files by name, for a host without a folder and for tests. */
        class InMemoryCeilingsSource final : public CeilingsSource
        {
            public:
                /** @brief Adds a file, or replaces the one of that name. */
                void put(std::string file_name, std::string text)
                {
                    files_.insert_or_assign(std::move(file_name), std::move(text));
                }

                [[nodiscard]] std::optional<std::string> read(const std::string& file_name) const override
                {
                    const auto found = files_.find(file_name);
                    if (found == files_.end())
                        return std::nullopt;
                    return found->second;
                }

                [[nodiscard]] std::string location(const std::string& file_name) const override
                {
                    return "memory:" + file_name;
                }

            private:
                /// @brief The texts, by file name.
                std::map<std::string, std::string> files_;
        };
    } // namespace Profiling
} // namespace SushiBLAS
