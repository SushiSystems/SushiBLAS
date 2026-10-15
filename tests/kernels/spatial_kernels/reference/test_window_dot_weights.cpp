/**************************************************************************/
/* test_window_dot_weights.cpp                                            */
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
 * @file test_window_dot_weights.cpp
 * @brief Tests that the reference weight gradient pairs with the window product as a bilinear form.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/window_dot.hpp>
#include <SushiBLAS/spatial_kernels/reference/window_dot_weights.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceWindowDotWeights, PairsWithTheWindowProduct)
        {
            // <window_dot(x, w), dy> equals <w, dweights(dy, x)>, exact on small integers.
            const Spatial::WindowSpan span = nhwc_span(2, 7, 6, 3, uneven_geometry(), 1);
            const Spatial::TapSpan taps{span.geometry.kernel_w * 3, 3, 1};
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> w = small_integers<double>(span.patch_elements(), 2);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 3);
            std::vector<double> y(static_cast<std::size_t>(span.output_elements()));
            std::vector<double> dw(static_cast<std::size_t>(span.patch_elements()));

            Kernels::Reference::window_dot(span, taps, x.data(), w.data(), y.data());
            Kernels::Reference::window_dot_weights_backward(span, taps, dy.data(), x.data(),
                                                            dw.data());

            EXPECT_EQ(std::inner_product(y.begin(), y.end(), dy.begin(), 0.0),
                      std::inner_product(w.begin(), w.end(), dw.begin(), 0.0));
        }

        TEST(Small_ReferenceWindowDotWeights, AOneTapWindowGivesTheChannelDotProduct)
        {
            const Spatial::WindowSpan span = nhwc_span(1, 1, 2, 1, Spatial::ConvGeometry{}, 0);
            const std::vector<float> x{2.0f, 3.0f};
            const std::vector<float> dy{5.0f, 7.0f};
            std::vector<float> dw(1, 0.0f);

            Kernels::Reference::window_dot_weights_backward(span, Spatial::TapSpan{1, 1, 1},
                                                            dy.data(), x.data(), dw.data());

            EXPECT_EQ(dw[0], 31.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
