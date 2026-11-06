/**************************************************************************/
/* gemm_epilogue.hpp                                                      */
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
 * @file gemm_epilogue.hpp
 * @brief Declares GemmIdentityEpilogue, the default GEMM epilogue.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Stores the computed GEMM element unchanged.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct GemmIdentityEpilogue
        {
            /**
             * @brief Returns value unchanged.
             * @param value The fully formed C element, alpha*(op(A)*op(B)) + beta*C.
             */
            template<typename T>
            [[nodiscard]] T operator()(T value,
                                       [[maybe_unused]] int64_t row,
                                       [[maybe_unused]] int64_t col) const noexcept
            {
                return value;
            }
        };

    } // namespace Kernels
} // namespace SushiBLAS
