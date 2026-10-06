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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
