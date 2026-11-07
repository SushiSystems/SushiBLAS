/**************************************************************************/
/* compile_blas_flags.cpp                                                 */
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
 * @file compile_blas_flags.cpp
 * @brief Checks at compile time what test_blas_flags.cpp cannot test at run time (REPORT.md
 *        decision 5).
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(transpose_of(true) == Core::Transpose::Trans, "Usable in a constant.");
        static_assert(side_of(false) == Core::Side::Right, "Usable in a constant.");
    } // namespace Test
} // namespace SushiBLAS
