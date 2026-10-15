/**************************************************************************/
/* test_channels.cpp                                                      */
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
 * @file test_channels.cpp
 * @brief Tests that the reference channel copy moves a range of channels between offsets.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/channels.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceChannels, CopiesTheRangeIntoItsOffsetAtEveryPixel)
        {
            // Input and output both hold three channels per pixel, two pixels.
            const Spatial::WindowSpan span = nhwc_span(1, 1, 2, 3, Spatial::ConvGeometry{}, 0);
            const std::vector<float> in{1, 2, 3, 4, 5, 6};
            std::vector<float> out(6, 0.0f);

            Kernels::Reference::copy_channel_range(span, 1, 0, 2, in.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{2, 3, 0, 5, 6, 0}));
        }
    } // namespace Test
} // namespace SushiBLAS
