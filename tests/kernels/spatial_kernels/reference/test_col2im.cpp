/**************************************************************************/
/* test_col2im.cpp                                                        */
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
 * @file test_col2im.cpp
 * @brief Tests that the reference col2im is the adjoint of im2col and counts the overlaps.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/col2im.hpp>
#include <SushiBLAS/spatial_kernels/reference/im2col.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferenceCol2Im, IsTheAdjointOfIm2Col)
        {
            // <im2col(x), y> equals <x, col2im(y)>; on small integers both sides are exact.
            const Spatial::WindowSpan span = nhwc_span(2, 6, 5, 2, uneven_geometry(), 1);
            const Spatial::ColumnSpan cols{span.patch_elements(), 1};
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> y = small_integers<double>(span.column_elements(), 2);
            std::vector<double> columns(static_cast<std::size_t>(span.column_elements()));
            std::vector<double> folded(static_cast<std::size_t>(span.input_elements()));

            Kernels::Reference::im2col(span, cols, x.data(), columns.data());
            Kernels::Reference::col2im(span, cols, y.data(), folded.data());

            EXPECT_EQ(std::inner_product(columns.begin(), columns.end(), y.begin(), 0.0),
                      std::inner_product(x.begin(), x.end(), folded.begin(), 0.0));
        }

        TEST(Unit_ReferenceCol2Im, OnesFoldIntoTheOverlapCountOfEachPixel)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 2;
            geometry.kernel_w = 2;
            const Spatial::WindowSpan span = nhwc_span(1, 3, 3, 1, geometry, 0);
            const std::vector<float> ones(static_cast<std::size_t>(span.column_elements()), 1.0f);
            std::vector<float> counts(9, 0.0f);

            Kernels::Reference::col2im(span, Spatial::ColumnSpan{4, 1}, ones.data(),
                                       counts.data());

            EXPECT_EQ(counts, (std::vector<float>{1, 2, 1, 2, 4, 2, 1, 2, 1}));
        }
    } // namespace Test
} // namespace SushiBLAS
