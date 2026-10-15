/**************************************************************************/
/* test_window.cpp                                                        */
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
 * @file test_window.cpp
 * @brief Tests the window, pooling, column and tap spans of a geometry and their refusals.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns a 3 x 3 window of stride 2 padded by 1 on the top and the left. */
        static Spatial::ConvGeometry strided_three()
        {
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.stride_h = 2;
            g.stride_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;
            return g;
        }

        /** @brief Builds NHWC tensors on the fixture's engine. */
        class Unit_Window : public SushiBLASTest
        {

        };

        TEST_F(Unit_Window, TheSpanOfADenseInputFollowsItsStrides)
        {
            const Tensor in = engine->create_tensor({2, 5, 6, 3});
            const Tensor out = engine->create_tensor({2, 3, 3, 3});

            const Spatial::WindowSpan span =
                Spatial::window_span_for(in, strided_three(), out, "conv");

            EXPECT_EQ(span.batch, int64_t{2});
            EXPECT_EQ(span.in_row_stride, int64_t{18});
            EXPECT_EQ(span.windows(), int64_t{18});
            EXPECT_EQ(span.patch_elements(), int64_t{27});
        }

        TEST_F(Unit_Window, AnInvalidGeometryOrAnOutputPastTheInputIsRefused)
        {
            const Tensor in = engine->create_tensor({1, 5, 5, 1});
            Spatial::ConvGeometry zero_stride = strided_three();
            zero_stride.stride_h = 0;

            EXPECT_THROW(Spatial::require_valid_geometry(zero_stride, "conv"),
                         std::invalid_argument);
            EXPECT_THROW((void)Spatial::window_span_for(in, strided_three(), 5, 1, "conv"),
                         std::invalid_argument);
        }

        TEST_F(Unit_Window, APoolingWindowOfOnlyPaddingIsRefused)
        {
            Spatial::ConvGeometry g;
            g.kernel_h = 1;
            g.kernel_w = 1;
            g.pad_top = 1;
            const Tensor in = engine->create_tensor({1, 2, 2, 1});
            const Tensor out = engine->create_tensor({1, 2, 2, 1});

            EXPECT_FALSE(Spatial::window_has_live_tap(-1, 1, 1, 2));
            EXPECT_THROW((void)Spatial::pooling_span_for(in, g, out, "pool"),
                         std::invalid_argument);
        }

        TEST_F(Unit_Window, ColumnAndTapSpansMatchTheirMatrices)
        {
            const Tensor in = engine->create_tensor({2, 5, 6, 3});
            const Tensor out = engine->create_tensor({2, 3, 3, 3});
            const Spatial::WindowSpan span =
                Spatial::window_span_for(in, strided_three(), out, "conv");

            EXPECT_NO_THROW((void)Spatial::column_span_for(engine->create_tensor({18, 27}), span,
                                                           "conv"));
            EXPECT_THROW((void)Spatial::column_span_for(engine->create_tensor({18, 26}), span,
                                                        "conv"),
                         std::invalid_argument);
            EXPECT_NO_THROW((void)Spatial::tap_span_for(engine->create_tensor({3, 3, 3}), span,
                                                        "conv", "taps"));
        }
    } // namespace Test
} // namespace SushiBLAS
