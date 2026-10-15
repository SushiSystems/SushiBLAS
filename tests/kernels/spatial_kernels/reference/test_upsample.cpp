/**************************************************************************/
/* test_upsample.cpp                                                      */
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
 * @file test_upsample.cpp
 * @brief Tests the reference nearest upsample against hand values and its backward as its adjoint.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/upsample.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns the span of an upsample of a dense NHWC input by scale_h x scale_w. */
        static Spatial::WindowSpan upsample_span(int64_t batch, int64_t height, int64_t width,
                                                 int64_t channels, int64_t scale_h,
                                                 int64_t scale_w)
        {
            Spatial::WindowSpan span = nhwc_span(batch, height, width, channels,
                                                 Spatial::ConvGeometry{}, 0);
            span.out_h = height * scale_h;
            span.out_w = width * scale_w;
            span.out_row_stride = span.out_w * channels;
            span.out_batch_stride = span.out_h * span.out_w * channels;
            return span;
        }

        TEST(Unit_ReferenceUpsample, RepeatsEachPixelOverItsBlock)
        {
            const Spatial::WindowSpan span = upsample_span(1, 1, 2, 1, 2, 2);
            const std::vector<float> in{1.0f, 2.0f};
            std::vector<float> out(8, 0.0f);

            Kernels::Reference::upsample_nearest(span, 2, 2, in.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{1, 1, 2, 2, 1, 1, 2, 2}));
        }

        TEST(Unit_ReferenceUpsample, TheBackwardIsTheAdjoint)
        {
            const Spatial::WindowSpan span = upsample_span(2, 3, 4, 2, 3, 2);
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            std::vector<double> y(static_cast<std::size_t>(span.output_elements()));
            std::vector<double> dx(static_cast<std::size_t>(span.input_elements()));

            Kernels::Reference::upsample_nearest(span, 3, 2, x.data(), y.data());
            Kernels::Reference::upsample_nearest_backward(span, 3, 2, dy.data(), dx.data());

            EXPECT_EQ(std::inner_product(y.begin(), y.end(), dy.begin(), 0.0),
                      std::inner_product(x.begin(), x.end(), dx.begin(), 0.0));
        }
    } // namespace Test
} // namespace SushiBLAS
