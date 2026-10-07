/**************************************************************************/
/* test_spatial_channels.cpp                                              */
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
 * @file test_spatial_channels.cpp
 * @brief Tests SpatialOps::concat_channels and slice_channels.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The bit pattern of a float; see test_spatial_pool_avg.cpp. */
            uint32_t channels_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialChannelsTest : public SushiBLASTest
        {
            protected:
                /** @brief A rank-4 NHWC float tensor holding @p values in memory order. */
                Tensor image(int64_t n, int64_t h, int64_t w, int64_t c,
                             std::initializer_list<float> values)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    int64_t i = 0;
                    for (float v : values) p[i++] = v;
                    return t;
                }

                /** @brief A rank-4 NHWC float tensor whose every element is @p value. */
                Tensor filled(int64_t n, int64_t h, int64_t w, int64_t c, float value)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i) p[i] = value;
                    return t;
                }

                /** @brief A rank-4 NHWC float tensor filled with a reproducible pattern. */
                Tensor patterned(int64_t n, int64_t h, int64_t w, int64_t c, int64_t seed)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::Float32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i)
                        p[i] = static_cast<float>(((i + seed) * 37) % 23) - 11.0f;
                    return t;
                }
        };

        TEST_F(Unit_SpatialChannelsTest, ConcatPlacesEachSourceAtItsOffset)
        {
            // Two pixels expose an ignored column stride. See tests/functional/unit/README.md.
            auto a = image(1, 1, 2, 2, {10, 11,
                                        12, 13});
            auto b = image(1, 1, 2, 3, {20, 21, 22,
                                        23, 24, 25});

            const float sentinel = -99.0f;
            auto dst = filled(1, 1, 2, 5, sentinel);

            SushiBLAS::SpatialOps(*engine).concat_channels(a, dst, 0);
            engine->execute();

            const float* got = dst.data_as<float>();

            // A's write must stop at channel 2. The sentinel is what says so.
            const std::vector<float> after_a = {10, 11, sentinel, sentinel, sentinel,
                                                12, 13, sentinel, sentinel, sentinel};
            for (std::size_t i = 0; i < after_a.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], after_a[i]) << "element " << i << " after the first concat";

            SushiBLAS::SpatialOps(*engine).concat_channels(b, dst, 2);
            engine->execute();

            const std::vector<float> expected = {10, 11, 20, 21, 22,
                                                 12, 13, 23, 24, 25};
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "element " << i << " after both concats";
        }

        TEST_F(Unit_SpatialChannelsTest, SliceIsConcatsInverse)
        {
            auto a = patterned(2, 3, 4, 2, 0);
            auto b = patterned(2, 3, 4, 5, 7);

            auto joined = filled(2, 3, 4, 7, -1.0f);
            SushiBLAS::SpatialOps(*engine).concat_channels(a, joined, 0);
            SushiBLAS::SpatialOps(*engine).concat_channels(b, joined, 2);

            auto a_back = filled(2, 3, 4, 2, 0.0f);
            auto b_back = filled(2, 3, 4, 5, 0.0f);
            SushiBLAS::SpatialOps(*engine).slice_channels(joined, 0, a_back);
            SushiBLAS::SpatialOps(*engine).slice_channels(joined, 2, b_back);
            engine->execute();

            const float* a_in = a.data_as<float>();
            const float* a_out = a_back.data_as<float>();
            for (int64_t i = 0; i < a.num_elements; ++i)
                EXPECT_EQ(channels_float_bits(a_out[i]), channels_float_bits(a_in[i]))
                    << "first source, element " << i;

            const float* b_in = b.data_as<float>();
            const float* b_out = b_back.data_as<float>();
            for (int64_t i = 0; i < b.num_elements; ++i)
                EXPECT_EQ(channels_float_bits(b_out[i]), channels_float_bits(b_in[i]))
                    << "second source, element " << i;
        }

        TEST_F(Unit_SpatialChannelsTest, TheyAreEachOthersBackward)
        {
            // Concat's gradient is a slice at the same offset. See tests/functional/unit/README.md.
            auto a = patterned(1, 2, 2, 3, 3);
            auto b = patterned(1, 2, 2, 4, 11);

            auto c = filled(1, 2, 2, 7, -1.0f);
            SushiBLAS::SpatialOps(*engine).concat_channels(a, c, 0);
            SushiBLAS::SpatialOps(*engine).concat_channels(b, c, 3);

            auto a_grad = filled(1, 2, 2, 3, 0.0f);
            SushiBLAS::SpatialOps(*engine).slice_channels(c, 0, a_grad);
            engine->execute();

            const float* expected = a.data_as<float>();
            const float* got = a_grad.data_as<float>();
            for (int64_t i = 0; i < a.num_elements; ++i)
                EXPECT_EQ(channels_float_bits(got[i]), channels_float_bits(expected[i]))
                    << "element " << i;
        }

        TEST_F(Unit_SpatialChannelsTest, ARangeThatRunsPastTheChannelCountIsRefused)
        {
            auto three = filled(1, 2, 2, 3, 1.0f);
            auto five = filled(1, 2, 2, 5, 0.0f);

            // Writing three channels starting at 3 needs six, and there are five.
            try
            {
                SushiBLAS::SpatialOps(*engine).concat_channels(three, five, 3);
                FAIL() << "a concat past the destination's channel count was accepted";
            }
            catch (const std::invalid_argument& e)
            {
                const std::string message = e.what();
                EXPECT_NE(message.find("spatial.concat_channels"), std::string::npos)
                    << "the message does not name the operation: " << message;
            }

            // Reading three channels starting at 4 needs seven, and there are five.
            auto destination = filled(1, 2, 2, 3, 0.0f);
            try
            {
                SushiBLAS::SpatialOps(*engine).slice_channels(five, 4, destination);
                FAIL() << "a slice past the source's channel count was accepted";
            }
            catch (const std::invalid_argument& e)
            {
                const std::string message = e.what();
                EXPECT_NE(message.find("spatial.slice_channels"), std::string::npos)
                    << "the message does not name the operation: " << message;
            }
        }

        TEST_F(Unit_SpatialChannelsTest, AStridedSourceIsAddressedThroughItsStrides)
        {
            // The view sees only the parent's even channels. See tests/functional/unit/README.md.
            auto parent = image(1, 2, 2, 4, {0, 1, 2, 3,      4, 5, 6, 7,
                                             8, 9, 10, 11,    12, 13, 14, 15});

            const std::vector<int64_t> dims = {1, 2, 2, 2};
            std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
            strides[0] = 16;   // one image
            strides[1] = 8;    // one row of two four-channel pixels
            strides[2] = 4;    // one four-channel pixel
            strides[3] = 2;    // every other channel

            const Tensor view(parent.storage, dims, strides, Core::DataType::Float32,
                              parent.storage_offset, parent.layout);

            auto dst = filled(1, 2, 2, 2, -1.0f);
            SushiBLAS::SpatialOps(*engine).concat_channels(view, dst, 0);
            engine->execute();

            const std::vector<float> expected = {0, 2,   4, 6,
                                                 8, 10,  12, 14};

            const float* got = dst.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "element " << i;
        }

        TEST_F(Unit_SpatialChannelsTest, TheSyclKernelAgreesWithTheReferenceKernel)
        {
            // The two forced modes are compared here. See tests/functional/unit/README.md.
            const int64_t N = 2, H = 3, W = 4, C = 3, WIDE = 7, OFFSET = 2;

            auto src = patterned(N, H, W, C, 5);

            auto from_reference = patterned(N, H, W, WIDE, 13);
            auto from_device = patterned(N, H, W, WIDE, 13);

            engine->set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(*engine).concat_channels(src, from_reference, OFFSET);
            engine->execute();

            engine->set_mode(ExecutionMode::ForceSycl);
            SushiBLAS::SpatialOps(*engine).concat_channels(src, from_device, OFFSET);
            engine->execute();

            const float* expected = from_reference.data_as<float>();
            const float* got = from_device.data_as<float>();
            for (int64_t i = 0; i < from_reference.num_elements; ++i)
                EXPECT_EQ(channels_float_bits(got[i]), channels_float_bits(expected[i]))
                    << "element " << i << ": reference " << expected[i]
                    << ", device " << got[i];

            auto sliced_reference = filled(N, H, W, C, 0.0f);
            auto sliced_device = filled(N, H, W, C, 0.0f);

            engine->set_mode(ExecutionMode::ForceReference);
            SushiBLAS::SpatialOps(*engine).slice_channels(from_reference, OFFSET, sliced_reference);
            engine->execute();

            engine->set_mode(ExecutionMode::ForceSycl);
            SushiBLAS::SpatialOps(*engine).slice_channels(from_reference, OFFSET, sliced_device);
            engine->execute();

            const float* expected_slice = sliced_reference.data_as<float>();
            const float* got_slice = sliced_device.data_as<float>();
            for (int64_t i = 0; i < sliced_reference.num_elements; ++i)
                EXPECT_EQ(channels_float_bits(got_slice[i]), channels_float_bits(expected_slice[i]))
                    << "slice element " << i;
        }

        TEST_F(Unit_SpatialChannelsTest, MalformedSpatialOperandsNameTheOperand)
        {
            /** @brief Runs a record call and expects std::invalid_argument naming op and operand. */
            const auto expect_named = [](const auto& record, const char* op, const char* operand)
            {
                try
                {
                    record();
                    ADD_FAILURE() << op << " accepted a malformed '" << operand << "'";
                }
                catch (const std::invalid_argument& e)
                {
                    const std::string message = e.what();
                    EXPECT_NE(message.find(op), std::string::npos)
                        << "the message does not name the operation: " << message;
                    EXPECT_NE(message.find(std::string("operand '") + operand + "'"),
                              std::string::npos)
                        << "the message does not name the operand: " << message;
                }
            };

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            SushiBLAS::SpatialOps spatial(*engine);

            auto rank3 = engine->create_tensor({1, 4, 4}, Core::DataType::Float32);
            auto pooled = filled(1, 2, 2, 1, 0.0f);
            expect_named([&] { spatial.pool_avg(rank3, pooled, g); }, "spatial.pool_avg", "in");

            auto picture = filled(1, 4, 4, 1, 1.0f);
            auto wide_cols = engine->create_tensor({4, 4}, Core::DataType::Float64);
            expect_named([&] { spatial.im2col(picture, wide_cols, g, 2, 2); },
                         "spatial.im2col", "cols");

            auto narrow_indices = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Int32);
            expect_named([&] { spatial.pool_max(picture, pooled, narrow_indices, g); },
                         "spatial.pool_max", "indices");

            auto three = filled(1, 2, 2, 3, 1.0f);
            auto five = filled(1, 2, 2, 5, 0.0f);
            expect_named([&] { spatial.concat_channels(three, five, -1); },
                         "spatial.concat_channels", "dst");
        }

    } // namespace Test
} // namespace SushiBLAS
