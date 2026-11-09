/**************************************************************************/
/* compile_blas.cpp                                                       */
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
 * @file compile_blas.cpp
 * @brief Checks at compile time that blas.hpp alone reaches a reference and a SYCL kernel of each BLAS level; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/blas_kernels/blas.hpp>

namespace SushiBLAS
{
    static_assert(std::is_function_v<decltype(Kernels::Reference::axpy<double>)>, "blas.hpp reaches the reference level 1");
    static_assert(std::is_function_v<decltype(Kernels::SYCL::axpy<double>)>, "blas.hpp reaches the SYCL level 1");
    static_assert(std::is_function_v<decltype(Kernels::Reference::gemv<double>)>, "blas.hpp reaches the reference level 2");
    static_assert(std::is_function_v<decltype(Kernels::SYCL::gemv<double>)>, "blas.hpp reaches the SYCL level 2");
    static_assert(std::is_function_v<decltype(Kernels::SYCL::syrk<double>)>, "blas.hpp reaches the SYCL level 3");
} // namespace SushiBLAS
