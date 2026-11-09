/**************************************************************************/
/* compile_symmetric_product.cpp                                          */
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
 * @file compile_symmetric_product.cpp
 * @brief Checks at compile time that the SYCL symmetric_product exists for a packed access and a symmetric element; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/blas_kernels/sycl/level2/symmetric_product.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>

namespace SushiBLAS
{
    static_assert(std::is_function_v<decltype(Kernels::SYCL::Detail::symmetric_product<
                      double, Kernels::Unconjugated, Kernels::PackedAccess>)>,
                  "symmetric_product is a function of the queue and the packed access");
} // namespace SushiBLAS
