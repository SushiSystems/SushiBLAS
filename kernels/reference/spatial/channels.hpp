/****************************************************************/
/* channels.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file channels.hpp
 * @brief Defines the reference kernel that copies a run of channels.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/engine/spatial/geometry.hpp>

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
