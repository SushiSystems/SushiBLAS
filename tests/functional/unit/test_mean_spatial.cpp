/**************************************************************************/
/* test_mean_spatial.cpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// `mean_spatial`, and the regression that the second segment axis it added did
// not disturb the reductions that have only ever had one.
//
// Every expectation is worked out here, in double, from the same generating rule
// the input was filled from - never from another path in this library. A
// differential oracle against `sum_along` over a reshaped view would agree with
// `mean_spatial` in exactly the cases where both read the strides wrongly.

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Copies a FLOAT32 tensor's elements back to the host. */
            std::vector<float> read_back(const Tensor& t)
            {
                const std::size_t n = static_cast<std::size_t>(t.num_elements);
                std::vector<float> host(n);
                auto& q = TestEnvironment::get_context().get_queue();
                q.memcpy(host.data(), t.data(), n * sizeof(float)).wait();
                return host;
            }
        }

        class Unit_MeanSpatialTest : public SushiBLASTest {};
        class Regression_AxisSpanTest : public SushiBLASTest {};

        // The shape the second segment axis exists for: one result per (n, c), whose
        // input offset is n * H*W*C + c and which no single segment stride reaches.
        TEST_F(Unit_MeanSpatialTest, FoldsTheTwoMiddleAxesOfAnNhwcTensor)
        {
            constexpr int64_t N = 2, H = 3, W = 4, C = 5;
            const std::size_t total = static_cast<std::size_t>(N * H * W * C);

            std::vector<float> src(total);
            for (std::size_t i = 0; i < total; ++i)
                src[i] = static_cast<float>(i % 17) - 8.0f;

            auto x = engine->create_tensor({N, H, W, C});
            auto out = engine->create_tensor({N, C});
            fill_tensor(x, src);

            engine->reductions().mean_spatial(x, out);
            engine->execute().wait();

            const std::vector<float> got = read_back(out);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(N * C));

            for (int64_t n = 0; n < N; ++n)
            {
                for (int64_t c = 0; c < C; ++c)
                {
                    double acc = 0.0;
                    for (int64_t h = 0; h < H; ++h)
                    {
                        for (int64_t w = 0; w < W; ++w)
                        {
                            const std::size_t idx =
                                static_cast<std::size_t>(((n * H + h) * W + w) * C + c);
                            acc += static_cast<double>(src[idx]);
                        }
                    }

                    const double expected = acc / static_cast<double>(H * W);
                    EXPECT_NEAR(got[static_cast<std::size_t>(n * C + c)],
                                static_cast<float>(expected), 1e-5f)
                        << "at (n, c) = (" << n << ", " << c << ")";
                }
            }
        }

        // N = 1 and C = 1 collapses both segment counts to one, which is the case a
        // generalisation most easily breaks: a stride that is never multiplied by
        // anything but zero is a stride nothing checks.
        TEST_F(Unit_MeanSpatialTest, ASingleSegmentPairIsStillCorrect)
        {
            auto x = engine->create_tensor({1, 2, 2, 1});
            auto out = engine->create_tensor({1, 1});
            fill_tensor(x, {1.0f, 2.0f, 3.0f, 4.0f});

            engine->reductions().mean_spatial(x, out);
            engine->execute().wait();

            const std::vector<float> got = read_back(out);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(1));
            EXPECT_NEAR(got[0], 2.5f, 1e-5f);
        }

        // One fold of 262144 elements: too few segments to fill the machine and a
        // segment far longer than a work-group, so `plan_axis` takes the cooperative
        // shape. Both shapes compute their segment's base address, and both had that
        // computation changed, so covering only the item-per-segment one would leave
        // half the change untested.
        TEST_F(Unit_MeanSpatialTest, ALongExtentReachesTheCooperativeShape)
        {
            constexpr int64_t H = 512, W = 512;
            const std::size_t total = static_cast<std::size_t>(H * W);

            std::vector<float> src(total);
            for (std::size_t i = 0; i < total; ++i)
                src[i] = static_cast<float>(i % 7) - 3.0f;

            auto x = engine->create_tensor({1, H, W, 1});
            auto out = engine->create_tensor({1, 1});
            fill_tensor(x, src);

            engine->reductions().mean_spatial(x, out);
            engine->execute().wait();

            double acc = 0.0;
            for (std::size_t i = 0; i < total; ++i) acc += static_cast<double>(src[i]);
            const double expected = acc / static_cast<double>(total);

            const std::vector<float> got = read_back(out);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(1));
            EXPECT_NEAR(got[0], static_cast<float>(expected), 1e-5f);
        }

        // Every existing caller aggregate-initialises AxisSpan with five values and so
        // now gets `segments2 = 1`, `segment_stride2 = 0` by default. If the fold index
        // is decomposed wrongly at that default, every row-wise reduction in the library
        // is wrong, and this is the cheapest place in the suite to see it.
        TEST_F(Regression_AxisSpanTest, ASpanWithNoSecondAxisReducesExactlyAsItAlwaysDid)
        {
            constexpr int64_t R = 6, K = 7;
            const std::size_t total = static_cast<std::size_t>(R * K);

            std::vector<float> src(total);
            for (std::size_t i = 0; i < total; ++i)
                src[i] = static_cast<float>(i);

            auto in = engine->create_tensor({R, K});
            auto out = engine->create_tensor({R});
            fill_tensor(in, src);

            engine->reductions().sum_along(in, 1, out);
            engine->execute().wait();

            const std::vector<float> got = read_back(out);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(R));

            for (int64_t r = 0; r < R; ++r)
            {
                double acc = 0.0;
                for (int64_t k = 0; k < K; ++k)
                    acc += static_cast<double>(src[static_cast<std::size_t>(r * K + k)]);

                EXPECT_NEAR(got[static_cast<std::size_t>(r)],
                            static_cast<float>(acc), 1e-5f) << "at row " << r;
            }
        }

    } // namespace Test
} // namespace SushiBLAS
