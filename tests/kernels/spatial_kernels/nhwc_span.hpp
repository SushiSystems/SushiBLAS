/**************************************************************************/
/* nhwc_span.hpp                                                          */
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
 * @file nhwc_span.hpp
 * @brief Declares nhwc_span, the window span of a dense NHWC input and output for a geometry.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/geometry/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Returns the span of a dense NHWC input and of the dense NHWC output a geometry
         *        gives it, padded by the geometry's top and left and by pad_after on the far sides.
         */
        [[nodiscard]] inline Spatial::WindowSpan nhwc_span(int64_t batch, int64_t height,
                                                           int64_t width, int64_t channels,
                                                           const Spatial::ConvGeometry& geometry,
                                                           int64_t pad_after)
        {
            Spatial::WindowSpan span;
            span.batch = batch;
            span.in_h = height;
            span.in_w = width;
            span.channels = channels;
            span.geometry = geometry;
            span.out_h = Spatial::output_extent(height, geometry.pad_top, pad_after,
                                                geometry.dilation_h, geometry.kernel_h,
                                                geometry.stride_h);
            span.out_w = Spatial::output_extent(width, geometry.pad_left, pad_after,
                                                geometry.dilation_w, geometry.kernel_w,
                                                geometry.stride_w);
            span.in_channel_stride = 1;
            span.in_col_stride = channels;
            span.in_row_stride = width * channels;
            span.in_batch_stride = height * width * channels;
            span.out_channel_stride = 1;
            span.out_col_stride = channels;
            span.out_row_stride = span.out_w * channels;
            span.out_batch_stride = span.out_h * span.out_w * channels;
            return span;
        }

        /** @brief Returns the geometry the device tests share: uneven kernel, stride, dilation. */
        [[nodiscard]] inline Spatial::ConvGeometry uneven_geometry() noexcept
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 3;
            geometry.kernel_w = 2;
            geometry.stride_h = 2;
            geometry.stride_w = 1;
            geometry.dilation_h = 1;
            geometry.dilation_w = 2;
            geometry.pad_top = 1;
            geometry.pad_left = 1;
            return geometry;
        }
    } // namespace Test
} // namespace SushiBLAS
