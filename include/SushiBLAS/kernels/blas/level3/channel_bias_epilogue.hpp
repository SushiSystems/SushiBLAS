/****************************************************************/
/* channel_bias_epilogue.hpp                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file channel_bias_epilogue.hpp
 * @brief Declares ChannelBiasEpilogue, the per-channel bias add for the GEMM epilogue.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Adds a per-channel bias to each GEMM output element, in either layout.
         * @warning The bias pointer must be device-accessible, outlive the submission and be a declared read.
         * @see include/SushiBLAS/kernels/README.md
         */
        struct ChannelBiasEpilogue
        {
            /** @brief The bias vector's first element, as the GEMM's own element type. */
            const void* bias = nullptr;

            /** @brief Bias elements to advance per output row; 0 when the channel is the column. */
            int64_t row_stride = 0;

            /** @brief Bias elements to advance per output column; 0 when the channel is the row. */
            int64_t col_stride = 1;

            /**
             * @brief Returns value plus the bias for its position.
             * @param value The fully formed C element, alpha*(op(A)*op(B)) + beta*C.
             */
            template<typename T>
            [[nodiscard]] T operator()(T value, int64_t row, int64_t col) const noexcept
            {
                return value + static_cast<const T*>(bias)[row * row_stride + col * col_stride];
            }
        };

    } // namespace Kernels
} // namespace SushiBLAS
