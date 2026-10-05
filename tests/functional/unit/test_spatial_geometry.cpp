/****************************************************************/
/* test_spatial_geometry.cpp                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// Device-free by construction: geometry.hpp includes <cstdint> and nothing
// else, so this whole file links no SYCL and needs no Engine. That is the
// property the header exists to have - SushiAI's host-lane shape inference
// shares this arithmetic and must not inherit the device stack to get it.

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
            // YOLOv3-tiny's 2x2 stride-1 max-pool, padded on the bottom and right
            // only, so 13x13 stays 13x13. This is the case CV-5 exists to make
            // ordinary, and it is the reason pad_bottom is a host-side number
            // rather than a ninth attribute slot.
            EXPECT_EQ(output_extent(13, 0, 1, 1, 2, 1), 13);
            // Symmetric padding of the same total would be a different answer,
            // which is the point: the two are not interchangeable.
            EXPECT_EQ(output_extent(13, 1, 0, 1, 2, 1), 13);
            EXPECT_EQ(output_extent(13, 0, 0, 1, 2, 1), 12);
        }

        TEST(Unit_SpatialGeometry, NonSquareAndAnisotropicGeometryIsExpressible)
        {
            // Height and width are independent all the way down, so a 1x7 kernel
            // with stride (1,2) is not a special case of anything.
            EXPECT_EQ(output_extent(20, 0, 0, 1, 1, 1), 20);
            EXPECT_EQ(output_extent(20, 3, 3, 1, 7, 2), 10);
        }

        TEST(Unit_SpatialGeometry, AWindowWiderThanThePaddedInputYieldsNoOutputRatherThanANegativeOne)
        {
            // The formula's numerator goes negative here. Integer division of a
            // negative numerator is implementation-flavoured and would produce a
            // garbage extent that later multiplies into an allocation size, so
            // it is answered explicitly.
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
            // A default-constructed geometry is 1x1, stride 1, no dilation, no
            // padding - so it maps every extent to itself. Anything that reads a
            // half-built geometry therefore gets a shape it can recognise as
            // wrong rather than a plausible one.
            const Spatial::ConvGeometry g;
            EXPECT_EQ(output_extent(17, 0, 0, g.dilation_h, g.kernel_h, g.stride_h), 17);
        }
    }
}
