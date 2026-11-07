/**************************************************************************/
/* sparse_conj_case.hpp                                                   */
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
 * @file sparse_conj_case.hpp
 * @brief Declares the hand-worked case that tells a conjugate transpose from a transpose on the
 *        4 x 5 sample matrix, for a real and for a complex element type.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <complex>
#include <SushiBLAS/core/traits.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Returns the one nonzero entry of the vector that selects row 0 of the sample.
         * @return 1 + i for a complex T, where row 0 stores 1 + 2i in column 0 and 2 - i in column 3;
         *         1 for a real T, where it stores 1 and 2.
         */
        template<typename T>
        [[nodiscard]] std::complex<double> conj_selector()
        {
            return Core::is_complex_v<T> ? std::complex<double>(1.0, 1.0)
                                         : std::complex<double>(1.0, 0.0);
        }

        /**
         * @brief Returns op(A) * s of the sample matrix A for the vector s that holds
         *        conj_selector() at element 0 and zeros elsewhere, five elements long.
         * @param conjugated Whether op is the conjugate transpose; a real T gives the same either way.
         */
        template<typename T>
        [[nodiscard]] std::vector<std::complex<double>> conj_probe_product([[maybe_unused]] bool conjugated)
        {
            const std::complex<double> zero(0.0, 0.0);
            if constexpr (Core::is_complex_v<T>)
            {
                // (1 + 2i)(1 + i) and (2 - i)(1 + i), or the same with each entry conjugated.
                return conjugated ? std::vector<std::complex<double>>{{3.0, -1.0}, zero, zero,
                                                                      {1.0, 3.0}, zero}
                                  : std::vector<std::complex<double>>{{-1.0, 3.0}, zero, zero,
                                                                      {3.0, 1.0}, zero};
            }
            else
            {
                return {{1.0, 0.0}, zero, zero, {2.0, 0.0}, zero};
            }
        }
    } // namespace Test
} // namespace SushiBLAS
