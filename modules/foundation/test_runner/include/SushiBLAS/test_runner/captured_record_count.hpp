/**************************************************************************/
/* captured_record_count.hpp                                              */
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
 * @file captured_record_count.hpp
 * @brief Declares count_records, which counts the records of one level that contain a text.
 * @author Mustafa Garip
 * @see modules/foundation/test_runner/README.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <SushiLog/log/log_capture.hpp>
#include <SushiLog/log/log_level.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns how many of @p records have @p level and a message containing @p text. */
        [[nodiscard]] inline std::size_t count_records(
            const std::vector<SushiLog::CapturedRecord>& records, SushiLog::LogLevel level,
            std::string_view text)
        {
            std::size_t found = 0;
            for (const SushiLog::CapturedRecord& record : records)
            {
                if (record.level == level && record.message.find(text) != std::string::npos)
                    ++found;
            }
            return found;
        }
    } // namespace Test
} // namespace SushiBLAS
