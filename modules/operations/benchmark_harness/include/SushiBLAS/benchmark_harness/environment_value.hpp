/**************************************************************************/
/* environment_value.hpp                                                  */
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
 * @file environment_value.hpp
 * @brief Declares environment_value, which reads an environment variable on every platform.
 * @author Mustafa Garip
 * @see modules/operations/benchmark_harness/README.md
 */

#pragma once

#include <cstdlib>
#include <string>

namespace SushiBLAS
{
    namespace Bench
    {
        /**
         * @brief Returns the value of the environment variable @p name.
         * @return The value; empty when the variable is unset or empty.
         */
        [[nodiscard]] inline std::string environment_value(const char* name)
        {
#if defined(_WIN32)
            char* buffer = nullptr;
            std::size_t length = 0;
            if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr)
                return {};
            std::string value(buffer);
            std::free(buffer);
            return value;
#else
            const char* value = std::getenv(name);
            return value == nullptr ? std::string() : std::string(value);
#endif
        }
    } // namespace Bench
} // namespace SushiBLAS
