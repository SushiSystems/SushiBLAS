/****************************************************************/
/* gemm_epilogue.hpp                                            */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
         * @see include/SushiBLAS/kernels/README.md
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
