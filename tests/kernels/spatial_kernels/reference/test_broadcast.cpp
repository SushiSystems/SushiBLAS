/**************************************************************************/
/* test_broadcast.cpp                                                     */
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
 * @file test_broadcast.cpp
 * @brief Tests the reference channel affine, spatial broadcast and layout conversion.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/broadcast.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceBroadcast, ChannelAffineScalesAndShiftsEachChannel)
        {
            const Spatial::WindowSpan span = nhwc_span(1, 1, 2, 2, Spatial::ConvGeometry{}, 0);
            const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
            const std::vector<float> gamma{10.0f, 100.0f};
            const std::vector<float> beta{1.0f, -1.0f};
            std::vector<float> out(4, 0.0f);

            Kernels::Reference::channel_affine(span, in.data(), gamma.data(), beta.data(), 1, 1,
                                               out.data());

            EXPECT_EQ(out, (std::vector<float>{11.0f, 199.0f, 31.0f, 399.0f}));
        }

        TEST(Small_ReferenceBroadcast, BroadcastSpatialFillsEveryPixelWithItsChannelValue)
        {
            const Spatial::WindowSpan span = nhwc_span(1, 2, 1, 2, Spatial::ConvGeometry{}, 0);
            const std::vector<float> in{3.0f, 5.0f};
            std::vector<float> out(4, 0.0f);

            Kernels::Reference::broadcast_spatial(span, in.data(), 2, 1, out.data(), 2.0f);

            EXPECT_EQ(out, (std::vector<float>{6.0f, 10.0f, 6.0f, 10.0f}));
        }

        TEST(Small_ReferenceBroadcast, LayoutConvertMovesNHWCToNCHW)
        {
            Spatial::WindowSpan span = nhwc_span(1, 1, 2, 3, Spatial::ConvGeometry{}, 0);
            // The output is NCHW: one plane of 1 x 2 pixels per channel.
            span.out_channel_stride = 2;
            span.out_col_stride = 1;
            span.out_row_stride = 2;
            span.out_batch_stride = 6;
            const std::vector<float> in{1, 2, 3, 4, 5, 6};
            std::vector<float> out(6, 0.0f);

            Kernels::Reference::layout_convert(span, in.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{1, 4, 2, 5, 3, 6}));
        }
    } // namespace Test
} // namespace SushiBLAS
