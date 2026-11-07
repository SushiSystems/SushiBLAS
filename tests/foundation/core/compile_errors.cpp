/**************************************************************************/
/* compile_errors.cpp                                                     */
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
 * @file compile_errors.cpp
 * @brief Checks at compile time what test_errors.cpp cannot test at run time (REPORT.md
 *        decision 5).
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <string>
#include <type_traits>
#include <SushiBLAS/core/errors.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_base_of<std::invalid_argument, UnsupportedExecution>::value,
                      "UnsupportedExecution derives from std::invalid_argument.");
        static_assert(std::is_final<UnsupportedExecution>::value,
                      "UnsupportedExecution is final.");
    } // namespace Test
} // namespace SushiBLAS
