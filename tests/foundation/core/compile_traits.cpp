/**************************************************************************/
/* compile_traits.cpp                                                     */
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
 * @file compile_traits.cpp
 * @brief Checks at compile time what test_traits.cpp cannot test at run time (REPORT.md
 *        decision 5).
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/traits.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(Core::is_complex_v<std::complex<float>>, "complex<float> is complex.");
        static_assert(Core::is_complex_v<std::complex<double>>, "complex<double> is complex.");
        static_assert(Core::is_complex<std::complex<float>>::value, "The trait agrees.");
        static_assert(!Core::is_complex_v<float>, "float is real.");
        static_assert(!Core::is_complex_v<double>, "double is real.");
        static_assert(!Core::is_complex_v<int32_t>, "int32_t is an integer.");
        static_assert(!Core::is_complex_v<int64_t>, "int64_t is an integer.");
        static_assert(!Core::is_complex_v<const std::complex<float>>,
                      "A qualified complex is not matched.");
    } // namespace Test
} // namespace SushiBLAS
