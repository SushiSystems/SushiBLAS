/**************************************************************************/
/* compile_common.cpp                                                     */
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
 * @file compile_common.cpp
 * @brief Checks at compile time what test_common.cpp cannot test at run time (REPORT.md
 *        decision 5).
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(static_cast<int>(Core::DataType::Half) == 0, "HALF is 0.");
        static_assert(static_cast<int>(Core::DataType::Float32) == 1, "FLOAT32 is 1.");
        static_assert(static_cast<int>(Core::DataType::Float64) == 2, "FLOAT64 is 2.");
        static_assert(static_cast<int>(Core::DataType::Complex32) == 3, "COMPLEX32 is 3.");
        static_assert(static_cast<int>(Core::DataType::Complex64) == 4, "COMPLEX64 is 4.");
        static_assert(static_cast<int>(Core::DataType::Int32) == 5, "INT32 is 5.");
        static_assert(static_cast<int>(Core::DataType::Int64) == 6, "INT64 is 6.");
        static_assert(static_cast<int>(Core::DataType::UInt32) == 7, "UINT32 is 7.");
        static_assert(Core::K_DATA_TYPE_COUNT == 8, "DataType has eight enumerators.");
        static_assert(sizeof(Core::DataType) == 1, "DataType is one byte.");
        static_assert(sizeof(Core::Layout) == 1, "Layout is one byte.");
    } // namespace Test
} // namespace SushiBLAS
