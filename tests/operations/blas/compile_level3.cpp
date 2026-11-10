/**************************************************************************/
/* compile_level3.cpp                                                     */
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
 * @file compile_level3.cpp
 * @brief Checks at compile time that BLASOps exposes the Level 3 facet and that only the facade builds it; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/blas/level3.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_base_of_v<Level3, BLASOps>, "BLASOps exposes the Level 3 facet.");
        static_assert(!std::is_constructible_v<Level3, TaskRecorder&>,
                      "Only the facade builds the Level 3 facet.");
    } // namespace Test
} // namespace SushiBLAS
