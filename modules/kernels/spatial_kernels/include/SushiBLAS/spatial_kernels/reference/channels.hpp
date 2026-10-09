/**************************************************************************/
/* channels.hpp                                                           */
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
 * @file channels.hpp
 * @brief Defines the reference kernel that copies a run of channels.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/support/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Copies a run of channels from one image into another.
             * @param span The grid and both operands' strides; `span.channels` is not read.
             * @param count How many channels to copy; it is not either tensor's channel count.
             * @see kernels/README.md
             */
            template<typename T>
            void copy_channel_range(const Spatial::WindowSpan& span, int64_t src_channel_offset,
                                    int64_t dst_channel_offset, int64_t count,
                                    const T* in, T* out)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t k = 0; k < count; ++k)
                            {
                                out[n * span.out_batch_stride
                                    + h * span.out_row_stride
                                    + w * span.out_col_stride
                                    + (dst_channel_offset + k) * span.out_channel_stride] =
                                    in[n * span.in_batch_stride
                                       + h * span.in_row_stride
                                       + w * span.in_col_stride
                                       + (src_channel_offset + k) * span.in_channel_stride];
                            }
                        }
                    }
                }
            }
        }
    }
}
