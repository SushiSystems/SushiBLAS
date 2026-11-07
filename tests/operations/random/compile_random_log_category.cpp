/**************************************************************************/
/* compile_random_log_category.cpp                                        */
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

/**
 * @file compile_random_log_category.cpp
 * @brief Checks that random_log_category.hpp compiles on its own and names K_RANDOM_LOG_CATEGORY; compiled, never run.
 * @author Mustafa Garip
 * @see tests/operations/random/README.md
 */

#include <SushiBLAS/random/random_log_category.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        [[maybe_unused]] constexpr const SushiLog::LogCategory& K_COMPILED_RANDOM_CATEGORY =
            K_RANDOM_LOG_CATEGORY;
    } // namespace Test
} // namespace SushiBLAS
