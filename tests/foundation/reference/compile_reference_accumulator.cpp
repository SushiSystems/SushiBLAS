/**************************************************************************/
/* compile_reference_accumulator.cpp                                      */
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
 * @file compile_reference_accumulator.cpp
 * @brief Checks at compile time the type a reference kernel accumulates each element type in.
 * @author Mustafa Garip
 */

#include <complex>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_same_v<Kernels::ReferenceAccumulator<sycl::half>, double>,
                      "HALF is summed in double.");
        static_assert(std::is_same_v<Kernels::ReferenceAccumulator<float>, double>,
                      "FLOAT32 is summed in double.");
        static_assert(std::is_same_v<Kernels::ReferenceAccumulator<double>, double>,
                      "FLOAT64 is summed in double.");
        static_assert(std::is_same_v<Kernels::ReferenceAccumulator<std::complex<float>>,
                                     std::complex<double>>,
                      "COMPLEX32 is summed in complex double.");
        static_assert(std::is_same_v<Kernels::ReferenceAccumulator<std::complex<double>>,
                                     std::complex<double>>,
                      "COMPLEX64 is summed in complex double.");
    } // namespace Test
} // namespace SushiBLAS
