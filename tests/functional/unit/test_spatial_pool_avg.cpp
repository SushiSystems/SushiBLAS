/****************************************************************/
/* test_spatial_pool_avg.cpp                                    */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
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
            /** @brief The bit pattern of a float; see test_spatial_im2col.cpp. */
            uint32_t pool_avg_float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialPoolAvgTest : public SushiBLASTest
        {
            protected:
                /** @brief A rank-4 NHWC float tensor holding @p values in memory order. */
                Tensor image(int64_t n, int64_t h, int64_t w, int64_t c,
                             std::initializer_list<float> values)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    int64_t i = 0;
                    for (float v : values) p[i++] = v;
                    return t;
                }

                /** @brief A rank-4 NHWC float tensor whose every element is @p value. */
                Tensor uniform(int64_t n, int64_t h, int64_t w, int64_t c, float value)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i) p[i] = value;
                    return t;
                }
        };

        // D-A4's pin. Without a padded case the convention is invisible: an
        // unpadded pool divides by the area and by the in-bounds count at once,
        // so every other test in this file passes under either rule.
        TEST_F(Unit_SpatialPoolAvgTest, TheAverageIsOverTheWholeWindowIncludingPaddedTaps)
        {
            auto in = image(1, 2, 2, 1, {4, 8,
                                         12, 16});

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::FLOAT32);
            engine->spatial().pool_avg(in, out, g);
            engine->execute();

            const float* got = out.data_as<float>();

            // The top-left window's only in-bounds tap is the element 4. Divided
            // by the window area that is 1; divided by the in-bounds count it
            // would be 4. That difference is the whole of count_include_pad.
            EXPECT_FLOAT_EQ(got[0], 1.0f) << "one real tap out of four, so 4/4";
            EXPECT_FLOAT_EQ(got[1], 3.0f) << "(4 + 8) / 4";
            EXPECT_FLOAT_EQ(got[2], 4.0f) << "(4 + 12) / 4";
            EXPECT_FLOAT_EQ(got[3], 10.0f) << "(4 + 8 + 12 + 16) / 4";
        }

        TEST_F(Unit_SpatialPoolAvgTest, AnUnpaddedPoolIsThePlainMean)
        {
            auto in = image(1, 4, 4, 1, {1, 3, 2, 0,
                                         4, 2, 1, 5,
                                         0, 7, 6, 2,
                                         8, 1, 3, 9});

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            auto out = engine->create_tensor({1, 2, 2, 1}, Core::DataType::FLOAT32);
            engine->spatial().pool_avg(in, out, g);
            engine->execute();

            const float* got = out.data_as<float>();
            EXPECT_FLOAT_EQ(got[0], 2.5f);   // (1+3+4+2)/4
            EXPECT_FLOAT_EQ(got[1], 2.0f);   // (2+0+1+5)/4
            EXPECT_FLOAT_EQ(got[2], 4.0f);   // (0+7+8+1)/4
            EXPECT_FLOAT_EQ(got[3], 5.0f);   // (6+2+3+9)/4
        }

        TEST_F(Unit_SpatialPoolAvgTest, TheBackwardSpreadsEachGradientOverItsWholeWindow)
        {
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            auto dy = image(1, 2, 2, 1, {4, 8,
                                         12, 16});
            auto dx = engine->create_tensor({1, 4, 4, 1}, Core::DataType::FLOAT32);

            engine->spatial().pool_avg_backward(dy, dx, g);
            engine->execute();

            // Non-overlapping quadrants, each receiving its window's dy/4.
            const std::vector<float> expected = {1, 1, 2, 2,
                                                 1, 1, 2, 2,
                                                 3, 3, 4, 4,
                                                 3, 3, 4, 4};

            const float* got = dx.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(got[i], expected[i]) << "pixel " << i;
        }

        TEST_F(Unit_SpatialPoolAvgTest, OverlappingWindowsSumTheirShares)
        {
            // 2x2 stride 1 over a 3x3 image: the centre pixel is covered by all
            // four windows and the corners by one each. A gather that stopped at
            // its first covering window would give the centre 0.25.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            auto dy = image(1, 2, 2, 1, {1, 2,
                                         3, 4});
            auto dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::FLOAT32);

            engine->spatial().pool_avg_backward(dy, dx, g);
            engine->execute();

            const float* got = dx.data_as<float>();
            EXPECT_FLOAT_EQ(got[0], 0.25f);                    // (0,0): window 0 only
            EXPECT_FLOAT_EQ(got[2], 0.5f);                     // (0,2): window 1 only
            EXPECT_FLOAT_EQ(got[4], (1 + 2 + 3 + 4) / 4.0f);   // (1,1): all four
            EXPECT_FLOAT_EQ(got[6], 0.75f);                    // (2,0): window 2 only
            EXPECT_FLOAT_EQ(got[8], 1.0f);                     // (2,2): window 3 only
        }

        TEST_F(Unit_SpatialPoolAvgTest, TheBackwardOfAConstantForwardIsUniform)
        {
            // A stride-1 pool padded so that every pixel is covered by exactly
            // R*S windows. Each contributes 1/(R*S), so every pixel must come
            // back exactly 1.0 - the closed-form check that the divisor and the
            // window set agree with each other. A divisor computed per-window
            // from the in-bounds count would make the border pixels larger.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            auto dy = uniform(1, 4, 4, 1, 1.0f);
            auto dx = engine->create_tensor({1, 3, 3, 1}, Core::DataType::FLOAT32);

            engine->spatial().pool_avg_backward(dy, dx, g);
            engine->execute();

            const float* got = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], 1.0f) << "pixel " << i;
        }

        TEST_F(Unit_SpatialPoolAvgTest, TheSyclKernelAgreesWithTheReferenceKernel)
        {
            // Compared here rather than under ExecutionMode::VERIFY, which runs
            // both kernels but lets the reference overwrite the device's answer
            // instead of checking it (see Internal::dispatch_with_mode).
            const int64_t N = 2, H = 7, W = 5, C = 3;

            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                p[i] = static_cast<float>((i * 37) % 23) - 11.0f;

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 1;
            g.dilation_h = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(H, 1, 1, 2, 3, 2);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 2, 1);

            auto from_reference = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::FLOAT32);
            auto from_device = engine->create_tensor({N, out_h, out_w, C}, Core::DataType::FLOAT32);

            engine->set_mode(ExecutionMode::FORCE_REFERENCE);
            engine->spatial().pool_avg(in, from_reference, g);
            engine->execute();

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            engine->spatial().pool_avg(in, from_device, g);
            engine->execute();

            const float* expected = from_reference.data_as<float>();
            const float* got = from_device.data_as<float>();
            for (int64_t i = 0; i < from_reference.num_elements; ++i)
                EXPECT_EQ(pool_avg_float_bits(got[i]), pool_avg_float_bits(expected[i]))
                    << "element " << i << ": reference " << expected[i]
                    << ", device " << got[i];
        }
    }
}
