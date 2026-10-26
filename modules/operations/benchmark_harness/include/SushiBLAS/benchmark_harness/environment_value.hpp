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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
