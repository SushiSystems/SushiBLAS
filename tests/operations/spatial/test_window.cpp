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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_window.cpp
 * @brief Tests the window, pooling, column and tap spans of a geometry and their refusals, on
 *        tensors that own no storage.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <vector>
#include <functional>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a dense float32 tensor of a shape that owns no storage. */
            Tensor shape_of(const std::vector<int64_t>& dims)
            {
                return Tensor::make_shape_only(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()));
            }

            /** @brief Returns a 3 x 3 window of stride 2 padded by 1 on the top and the left. */
            Spatial::ConvGeometry strided_three()
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

            /** @brief Returns the span of a 2 x 5 x 6 x 3 input under strided_three(). */
            Spatial::WindowSpan dense_span()
            {
                return Spatial::window_span_for(shape_of({2, 5, 6, 3}), strided_three(),
                                                shape_of({2, 3, 3, 3}), "conv");
            }

            /** @brief Expects check to throw std::invalid_argument with exactly this message. */
            void expect_refusal(const std::function<void()>& check, const std::string& message)
            {
                try
                {
                    check();
                    ADD_FAILURE() << "the check accepted the operands";
                }
                catch (const std::invalid_argument& error)
                {
                    EXPECT_EQ(std::string(error.what()), message);
                }
            }
        } // namespace

        TEST(Small_WindowSpan, TheSpanOfADenseInputFollowsItsStrides)
        {
            const Spatial::WindowSpan span = dense_span();

            EXPECT_EQ(span.batch, int64_t{2});
            EXPECT_EQ(span.in_row_stride, int64_t{18});
            EXPECT_EQ(span.windows(), int64_t{18});
            EXPECT_EQ(span.patch_elements(), int64_t{27});
        }

        TEST(Small_WindowSpan, AnOutputPastTheInputIsRefused)
        {
            const Tensor in = shape_of({1, 5, 5, 1});

            expect_refusal(
                [&] { (void)Spatial::window_span_for(in, strided_three(), 5, 1, "conv"); },
                "conv: operand 'in' has 5 rows padded by 1 above, but output row 4 would start "
                "at input row 7");
        }

        TEST(Small_WindowGeometry, AZeroStrideIsRefusedNamingTheField)
        {
            Spatial::ConvGeometry zero_stride = strided_three();
            zero_stride.stride_h = 0;

            expect_refusal([&] { Spatial::require_valid_geometry(zero_stride, "conv"); },
                           "conv: operand 'geometry' has stride_h = 0; it must be at least 1");
        }

        TEST(Small_PoolingSpan, APoolingWindowOfOnlyPaddingIsRefused)
        {
            Spatial::ConvGeometry g;
            g.pad_top = 1;
            const Tensor in = shape_of({1, 2, 2, 1});
            const Tensor out = shape_of({1, 2, 2, 1});

            expect_refusal([&] { (void)Spatial::pooling_span_for(in, g, out, "pool"); },
                           "pool: operand 'geometry' puts output row 0 entirely in the padding "
                           "(padding 1, kernel 1, dilation 1, input extent 2); a pooling window "
                           "needs a real tap");
        }

        TEST(Small_WindowLiveTap, AWindowWhoseOnlyTapIsBeforeTheInputHasNoLiveTap)
        {
            EXPECT_FALSE(Spatial::window_has_live_tap(-1, 1, 1, 2));
        }

        TEST(Small_WindowLiveTap, AWindowWhoseSecondTapReachesTheInputHasALiveTap)
        {
            EXPECT_TRUE(Spatial::window_has_live_tap(-1, 1, 2, 2));
        }

        TEST(Small_ColumnSpan, AMatrixOfTheWindowsByThePatchIsAccepted)
        {
            const Spatial::ColumnSpan columns =
                Spatial::column_span_for(shape_of({18, 27}), dense_span(), "conv");

            EXPECT_EQ(columns.row_stride, int64_t{27});
            EXPECT_EQ(columns.element_stride, int64_t{1});
        }

        TEST(Small_ColumnSpan, AMatrixOneColumnShortOfThePatchIsRefused)
        {
            const Spatial::WindowSpan span = dense_span();
            const Tensor cols = shape_of({18, 26});

            expect_refusal([&] { (void)Spatial::column_span_for(cols, span, "conv"); },
                           "conv: operand 'cols' has 26 columns, expected 27");
        }

        TEST(Small_TapSpan, ATapTensorOfTheKernelByTheChannelsIsAccepted)
        {
            const Spatial::TapSpan taps =
                Spatial::tap_span_for(shape_of({3, 3, 3}), dense_span(), "conv", "taps");

            EXPECT_EQ(taps.row_stride, int64_t{9});
            EXPECT_EQ(taps.col_stride, int64_t{3});
            EXPECT_EQ(taps.channel_stride, int64_t{1});
        }
    } // namespace Test
} // namespace SushiBLAS
