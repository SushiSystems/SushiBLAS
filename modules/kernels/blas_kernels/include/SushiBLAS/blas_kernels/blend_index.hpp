/**************************************************************************/
/* blend_index.hpp                                                        */
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
 * @file blend_index.hpp
 * @brief Declares the beta blend of an updated value with its prior value, shared by the SYCL matrix-vector and rank-k kernels.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /**
             * @brief Returns `scaled`, plus `beta * prior` unless beta is exactly zero.
             * @tparam W The type the sum is held in.
             * @tparam WS The type beta is multiplied in: W, or its real component for HERK.
             * @param prior The value being updated; a zero beta never reads it, so it may be
             *        uninitialised.
             */
            template<typename W, typename WS, typename Scalar, typename T>
            [[nodiscard]] constexpr W blend_with_prior(W scaled, Scalar beta, const T& prior) noexcept
            {
                return beta == Scalar(0) ? scaled : W(scaled + multiply(WS(beta), W(prior)));
            }
        }
    }
}
