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
#include <functional>
#include <optional>
#include <string>

namespace SushiBLAS
{
    namespace Bench
    {
        /**
         * @brief Looks a variable up by name.
         * @return The value, or an empty optional when the variable is unset.
         */
        using EnvironmentLookup = std::function<std::optional<std::string>(const char*)>;

        /** @brief Looks @p name up in the process environment; an empty optional when it is unset. */
        [[nodiscard]] inline std::optional<std::string> process_environment(const char* name)
        {
#if defined(_WIN32)
            char* buffer = nullptr;
            std::size_t length = 0;
            if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr)
                return std::nullopt;
            std::string value(buffer);
            std::free(buffer);
            return value;
#else
            const char* value = std::getenv(name);
            if (value == nullptr)
                return std::nullopt;
            return std::string(value);
#endif
        }

        /**
         * @brief Returns the value of the environment variable @p name.
         * @param lookup Where the variable is read from; the process environment by default.
         * @return The value; empty when the variable is unset or empty.
         */
        [[nodiscard]] inline std::string environment_value(const char* name,
                                                           const EnvironmentLookup& lookup = process_environment)
        {
            return lookup(name).value_or(std::string());
        }
    } // namespace Bench
} // namespace SushiBLAS
