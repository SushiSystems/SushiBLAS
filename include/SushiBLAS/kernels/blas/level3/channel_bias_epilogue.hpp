/**************************************************************************/
/* channel_bias_epilogue.hpp                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file channel_bias_epilogue.hpp
 * @brief A per-channel bias add for the GEMM epilogue, with the channel axis
 * chosen by a stride pair rather than baked into the type.
 *
 * The stride pair exists because "which axis is the channel" is a property of
 * the *tensor layout*, not of the bias operation. In NHWC the convolution GEMM
 * is `[N*H*W, C_in*R*S] x [C_in*R*S, C_out]`, so the channel index is the
 * **column** of C - which means the row-vector broadcast the existing
 * `Ops::GEMM_BIAS` family already performs applies to convolution verbatim,
 * with no new arithmetic. In NCHW the same convolution transposes and the
 * channel becomes the **row**. Encoding that as `(row_stride, col_stride)`
 * makes the second layout cost two integers on the host instead of a second
 * epilogue type, a second fused lowering recorder and a second set of tests.
 *
 * It lives under `include/` rather than the private repo-root `kernels/` tree
 * for the reason `kernels/blas.hpp` states: `Dispatch::execute_gemm`
 * (<SushiBLAS/engine/blas/dispatch.hpp>) instantiates the epilogue in a
 * *consumer's* translation unit, and a template a public seam instantiates
 * cannot stay private.
 */

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
        * @struct ChannelBiasEpilogue
        * @brief Adds a per-channel bias to each GEMM output element, in either layout.
        *
        * `value + bias[row * row_stride + col * col_stride]`, with the stride
        * pair set on the host. `(0, 1)` makes the **column** the channel, which
        * is what an NHWC convolution GEMM needs and what `Ops::GEMM_BIAS`'s
        * existing row-vector broadcast already does; `(1, 0)` makes the **row**
        * the channel, which is what an NCHW one would need. Neither is baked in,
        * so a second layout costs two numbers rather than a second epilogue and a
        * second fused lowering recorder.
        *
        * @warning The pointer is captured into a SYCL kernel. It must be
        *          device-accessible, must outlive the submission, and the access
        *          set that reads it has to be declared on the task -
        *          `Dispatch::gemm_epilogue(f).reads(bias_tensor)` is how, and
        *          `execute_gemm` cannot discover it (see its documentation).
        *
        * @see Kernels::GemmIdentityEpilogue for the no-op this degenerates to at a
        *      zero bias, and SushiBLAS::Dispatch::execute_gemm for the fusion
        *      contract and the host-readability constraint FORCE_REFERENCE adds.
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
            * @brief Returns @p value plus the bias for its position.
            *
            * @tparam T The element type of C.
            * @param value The fully formed C element, alpha*(op(A)*op(B)) + beta*C.
            * @param row The element's row in C.
            * @param col The element's column in C.
            * @return @p value plus the corresponding bias element.
            */
            template<typename T>
            [[nodiscard]] T operator()(T value, int64_t row, int64_t col) const noexcept
            {
                return value + static_cast<const T*>(bias)[row * row_stride + col * col_stride];
            }
        };

    } // namespace Kernels
} // namespace SushiBLAS
