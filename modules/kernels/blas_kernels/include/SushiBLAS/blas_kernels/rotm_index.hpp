/**************************************************************************/
/* rotm_index.hpp                                                         */
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
 * @file rotm_index.hpp
 * @brief Declares the flag test and the pair transformation of ROTM, shared by the reference and the SYCL kernel.
 * @author Mustafa Garip
 */

#pragma once

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Holds a transformed pair of ROTM. */
            template<typename W>
            struct RotmPair
            {
                W x;    /**< The new x component. */
                W y;    /**< The new y component. */
            };

            /** @brief Returns whether a ROTM flag selects the identity, which leaves x and y alone. */
            template<typename W>
            [[nodiscard]] constexpr bool rotm_is_identity(W flag) noexcept
            {
                return flag == W(-2);
            }

            /**
             * @brief Applies the ROTM matrix H the flag selects to the pair `(w, z)`.
             * @param flag -1 for the full H, 0 for unit diagonal, 1 for unit off-diagonal.
             * @pre The flag is not the identity flag.
             */
            template<typename W>
            [[nodiscard]] constexpr RotmPair<W> rotm_apply(W flag, W h11, W h21, W h12, W h22, W w,
                                                           W z) noexcept
            {
                if (flag < W(0))
                    return RotmPair<W>{w * h11 + z * h12, w * h21 + z * h22};
                if (flag == W(0))
                    return RotmPair<W>{w + z * h12, w * h21 + z};
                return RotmPair<W>{w * h11 + z, -w + z * h22};
            }
        }
    }
}
