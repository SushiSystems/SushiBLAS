/**************************************************************************/
/* test_pool_max.cpp                                                      */
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
 * @file test_pool_max.cpp
 * @brief Tests the reference max pool, the index it records and the backward that uses it.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_max.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ReferencePoolMax, KeepsTheLargestTapAndItsFlatPosition)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 2;
            geometry.kernel_w = 2;
            geometry.stride_h = 2;
            geometry.stride_w = 2;
            const Spatial::WindowSpan span = nhwc_span(1, 2, 4, 1, geometry, 0);
            const std::vector<float> in{1, 9, 3, 4, 5, 6, 7, 8};
            std::vector<float> out(2, 0.0f);
            std::vector<int64_t> indices(2, -1);

            Kernels::Reference::pool_max(span, in.data(), out.data(), indices.data());

            EXPECT_EQ(out, (std::vector<float>{9.0f, 8.0f}));
            EXPECT_EQ(indices, (std::vector<int64_t>{1, 7}));
        }

        TEST(Unit_ReferencePoolMax, TheBackwardSendsEachGradientToItsArgmax)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 2;
            geometry.kernel_w = 2;
            geometry.stride_h = 2;
            geometry.stride_w = 2;
            const Spatial::WindowSpan span = nhwc_span(1, 2, 4, 1, geometry, 0);
            const std::vector<float> dy{10.0f, 20.0f};
            const std::vector<int64_t> indices{1, 7};
            std::vector<float> dx(8, -1.0f);

            Kernels::Reference::pool_max_backward(span, dy.data(), indices.data(), dx.data());

            EXPECT_EQ(dx, (std::vector<float>{0, 10, 0, 0, 0, 0, 0, 20}));
        }
    } // namespace Test
} // namespace SushiBLAS
