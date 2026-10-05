/****************************************************************/
/* test_spatial_geometry.cpp                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_spatial_geometry.cpp
 * @brief Tests output_extent, the window span and the default ConvGeometry without a device.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Spatial::output_extent;

        TEST(Unit_SpatialGeometry, TheOrdinaryCasesMatchTheFormula)
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

        TEST(Unit_SpatialGeometry, DilationWidensTheKernelWithoutWideningTheOutput)
        {
            // A dilated 3x3 covers 5 rows, so "same" needs 2 either side.
            EXPECT_EQ(output_extent(32, 2, 2, 2, 3, 1), 32);
            EXPECT_EQ(output_extent(32, 0, 0, 2, 3, 1), 28);
            EXPECT_EQ(output_extent(32, 0, 0, 4, 3, 1), 24);
        }

        TEST(Unit_SpatialGeometry, AsymmetricPaddingIsWhatHoldsYoloTinysLastPool)
        {
            // Bottom and right padding keep 13x13 at 13x13. See tests/functional/unit/README.md.
            EXPECT_EQ(output_extent(13, 0, 1, 1, 2, 1), 13);
            // Padding on the other side is a separate case, not interchangeable with this one.
            EXPECT_EQ(output_extent(13, 1, 0, 1, 2, 1), 13);
            EXPECT_EQ(output_extent(13, 0, 0, 1, 2, 1), 12);
        }

        TEST(Unit_SpatialGeometry, NonSquareAndAnisotropicGeometryIsExpressible)
        {
            // Height and width are independent, so a 1x7 kernel at stride (1,2) is no special case.
            EXPECT_EQ(output_extent(20, 0, 0, 1, 1, 1), 20);
            EXPECT_EQ(output_extent(20, 3, 3, 1, 7, 2), 10);
        }

        TEST(Unit_SpatialGeometry, AWindowWiderThanThePaddedInputYieldsNoOutputRatherThanANegativeOne)
        {
            // A negative numerator is answered with zero. See tests/functional/unit/README.md.
            EXPECT_EQ(output_extent(3, 0, 0, 1, 5, 1), 0);
            EXPECT_EQ(output_extent(3, 0, 0, 4, 3, 1), 0);
        }

        TEST(Unit_SpatialGeometry, WindowSpanDerivesItsOwnCounts)
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

        TEST(Unit_SpatialGeometry, TheDefaultGeometryIsTheIdentityWindow)
        {
            // A default geometry maps every extent to itself. See tests/functional/unit/README.md.
            const Spatial::ConvGeometry g;
            EXPECT_EQ(output_extent(17, 0, 0, g.dilation_h, g.kernel_h, g.stride_h), 17);
        }
    }
}
