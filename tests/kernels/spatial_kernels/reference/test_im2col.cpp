/**************************************************************************/
/* test_im2col.cpp                                                        */
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
 * @file test_im2col.cpp
 * @brief Tests the reference im2col on a one-tap window and on a padded two-by-two window.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/im2col.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferenceIm2Col, AOneTapWindowCopiesTheInput)
        {
            const Spatial::WindowSpan span = nhwc_span(1, 2, 3, 2, Spatial::ConvGeometry{}, 0);
            const std::vector<float> in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            std::vector<float> columns(in.size(), 0.0f);

            Kernels::Reference::im2col(span, Spatial::ColumnSpan{2, 1}, in.data(),
                                       columns.data());

            EXPECT_EQ(columns, in);
        }

        TEST(Unit_ReferenceIm2Col, APaddedWindowReadsZerosOutsideTheImage)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 2;
            geometry.kernel_w = 2;
            geometry.pad_top = 1;
            geometry.pad_left = 1;
            const Spatial::WindowSpan span = nhwc_span(1, 2, 2, 1, geometry, 0);
            const std::vector<float> in{1, 2, 3, 4};
            std::vector<float> columns(static_cast<std::size_t>(span.column_elements()), -1.0f);

            Kernels::Reference::im2col(span, Spatial::ColumnSpan{4, 1}, in.data(),
                                       columns.data());

            // Two output positions per axis; each row is one window, taps in row order.
            EXPECT_EQ(columns, (std::vector<float>{0, 0, 0, 1, 0, 0, 1, 2,
                                                   0, 1, 0, 3, 1, 2, 3, 4}));
        }
    } // namespace Test
} // namespace SushiBLAS
