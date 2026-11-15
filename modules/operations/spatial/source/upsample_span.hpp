/**************************************************************************/
/* upsample_span.hpp                                                      */
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
 * @file upsample_span.hpp
 * @brief Declares the span builder that both upsampling operations share.
 * @author Mustafa Garip
 * @see modules/operations/spatial/README.md
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Detail
    {
        /**
         * @brief Builds the span for an upsample from its two rank-4 operands.
         * @return The span, in element counts, with both operands' strides filled in.
         * @throws std::invalid_argument when a rank is not 4, a scale is not positive, or the
         *         extents are not the exact scaled multiples.
         * @see modules/operations/spatial/README.md
         */
        inline Spatial::WindowSpan upsample_span_for(const Tensor& in, const Tensor& out,
                                                     int64_t scale_h, int64_t scale_w,
                                                     const char* name)
        {
            Dispatch::require_rank(in, 4, name, "in");
            Dispatch::require_rank(out, 4, name, "out");
            Dispatch::require_operand(scale_h > 0, name, "scale_h",
                                      "is {}; a scale must be positive", scale_h);
            Dispatch::require_operand(scale_w > 0, name, "scale_w",
                                      "is {}; a scale must be positive", scale_w);

            Spatial::WindowSpan span;
            span.batch = in.shape[0];
            span.in_h = in.shape[1];
            span.in_w = in.shape[2];
            span.channels = in.shape[3];
            span.out_h = out.shape[1];
            span.out_w = out.shape[2];

            Dispatch::require_extent(out.shape[0], span.batch, "images", name, "out");
            Dispatch::require_extent(out.shape[3], span.channels, "channels", name, "out");
            Dispatch::require_extent(span.out_h, span.in_h * scale_h, "rows", name, "out");
            Dispatch::require_extent(span.out_w, span.in_w * scale_w, "columns", name, "out");

            span.in_batch_stride = in.strides[0];
            span.in_row_stride = in.strides[1];
            span.in_col_stride = in.strides[2];
            span.in_channel_stride = in.strides[3];

            span.out_batch_stride = out.strides[0];
            span.out_row_stride = out.strides[1];
            span.out_col_stride = out.strides[2];
            span.out_channel_stride = out.strides[3];

            return span;
        }
    } // namespace Detail
} // namespace SushiBLAS
