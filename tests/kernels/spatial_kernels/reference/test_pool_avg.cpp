/**************************************************************************/
/* test_pool_avg.cpp                                                      */
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
 * @file test_pool_avg.cpp
 * @brief Tests the reference average pool against hand values and its backward as its adjoint.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_avg.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferencePoolAvg, AveragesEachWindowOverItsFullArea)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 2;
            geometry.kernel_w = 2;
            geometry.stride_h = 2;
            geometry.stride_w = 2;
            const Spatial::WindowSpan span = nhwc_span(1, 2, 4, 1, geometry, 0);
            const std::vector<float> in{1, 2, 3, 4, 5, 6, 7, 8};
            std::vector<float> out(2, 0.0f);

            Kernels::Reference::pool_avg(span, in.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{3.5f, 5.5f}));
        }

        TEST(Small_ReferencePoolAvg, TheBackwardIsTheAdjoint)
        {
            // A window area of 6 makes each average a sixth; the dot products agree to rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 7, 6, 2, uneven_geometry(), 1);
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            std::vector<double> y(static_cast<std::size_t>(span.output_elements()));
            std::vector<double> dx(static_cast<std::size_t>(span.input_elements()));

            Kernels::Reference::pool_avg(span, x.data(), y.data());
            Kernels::Reference::pool_avg_backward(span, dy.data(), dx.data());

            EXPECT_NEAR(std::inner_product(y.begin(), y.end(), dy.begin(), 0.0),
                        std::inner_product(x.begin(), x.end(), dx.begin(), 0.0), 1e-9);
        }
    } // namespace Test
} // namespace SushiBLAS
