/**************************************************************************/
/* test_spatial_geometry.cpp                                              */
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
 * @file test_spatial_geometry.cpp
 * @brief Tests output_extent, the window span and the default ConvGeometry without a device.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Spatial::output_extent;

        TEST(Medium_SpatialGeometry, TheOrdinaryCasesMatchTheFormula)
        {
            // ResNet-18's stem: 224 -> 112, 7x7 stride 2 pad 3.
            EXPECT_EQ(output_extent(224, 3, 3, 1, 7, 2), 112);
            // A 3x3 stride-1 "same" convolution keeps its extent.
            EXPECT_EQ(output_extent(32, 1, 1, 1, 3, 1), 32);
            // Stride 2 halves it, rounding down.
            EXPECT_EQ(output_extent(32, 1, 1, 1, 3, 2), 16);
            // No padding shrinks by the kernel.
            EXPECT_EQ(output_extent(32, 0, 0, 1, 3, 1), 30);
        }

        TEST(Medium_SpatialGeometry, DilationWidensTheKernelWithoutWideningTheOutput)
        {
            // A dilated 3x3 covers 5 rows, so "same" needs 2 either side.
            EXPECT_EQ(output_extent(32, 2, 2, 2, 3, 1), 32);
            EXPECT_EQ(output_extent(32, 0, 0, 2, 3, 1), 28);
            EXPECT_EQ(output_extent(32, 0, 0, 4, 3, 1), 24);
        }

        TEST(Medium_SpatialGeometry, AsymmetricPaddingIsWhatHoldsYoloTinysLastPool)
        {
            // Bottom and right padding keep 13x13 at 13x13. See the test README.
            EXPECT_EQ(output_extent(13, 0, 1, 1, 2, 1), 13);
            // Padding on the other side is a separate case, not interchangeable with this one.
            EXPECT_EQ(output_extent(13, 1, 0, 1, 2, 1), 13);
            EXPECT_EQ(output_extent(13, 0, 0, 1, 2, 1), 12);
        }

        TEST(Medium_SpatialGeometry, NonSquareAndAnisotropicGeometryIsExpressible)
        {
            // Height and width are independent, so a 1x7 kernel at stride (1,2) is no special case.
            EXPECT_EQ(output_extent(20, 0, 0, 1, 1, 1), 20);
            EXPECT_EQ(output_extent(20, 3, 3, 1, 7, 2), 10);
        }

        TEST(Medium_SpatialGeometry, AWindowWiderThanThePaddedInputYieldsNoOutputRatherThanANegativeOne)
        {
            // A negative numerator is answered with zero. See tests/foundation/geometry/README.md.
            EXPECT_EQ(output_extent(3, 0, 0, 1, 5, 1), 0);
            EXPECT_EQ(output_extent(3, 0, 0, 4, 3, 1), 0);
        }

        TEST(Medium_SpatialGeometry, WindowSpanDerivesItsOwnCounts)
        {
            Spatial::WindowSpan span;
            span.batch = 2;
            span.channels = 3;
            span.out_h = 4;
            span.out_w = 5;
            span.geometry.kernel_h = 3;
            span.geometry.kernel_w = 3;

            EXPECT_EQ(span.windows(), 2 * 4 * 5);
            EXPECT_EQ(span.patch_elements(), 3 * 3 * 3);
            EXPECT_EQ(span.column_elements(), 2 * 4 * 5 * 3 * 3 * 3);
        }

        TEST(Medium_SpatialGeometry, TheDefaultGeometryIsTheIdentityWindow)
        {
            // A default geometry maps every extent to itself. See the test README.
            const Spatial::ConvGeometry g;
            EXPECT_EQ(output_extent(17, 0, 0, g.dilation_h, g.kernel_h, g.stride_h), 17);
        }
    }
}
