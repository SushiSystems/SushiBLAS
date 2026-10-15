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
 * @file test_mean_spatial.cpp
 * @brief Tests ReductionOps::mean_spatial and the single-axis reductions that share its span.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
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
                auto& q = TestEnvironment::queue();
                q.memcpy(host.data(), t.data(), n * sizeof(float)).wait();
                return host;
            }
        }

        class Unit_MeanSpatialTest : public SushiBLASTest {};
        class Regression_AxisSpanTest : public SushiBLASTest {};

        /**
         * @brief Tests that mean_spatial folds the two middle axes into one result per (n, c).
         * @see tests/unit/README.md
         */
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

            SushiBLAS::ReductionOps(*engine).mean_spatial(x, out);
            engine->execute().value();

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

        /**
         * @brief Tests that a tensor with N = 1 and C = 1 still reduces correctly.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_MeanSpatialTest, ASingleSegmentPairIsStillCorrect)
        {
            auto x = engine->create_tensor({1, 2, 2, 1});
            auto out = engine->create_tensor({1, 1});
            fill_tensor(x, {1.0f, 2.0f, 3.0f, 4.0f});

            SushiBLAS::ReductionOps(*engine).mean_spatial(x, out);
            engine->execute().value();

            const std::vector<float> got = read_back(out);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(1));
            EXPECT_NEAR(got[0], 2.5f, 1e-5f);
        }

        /**
         * @brief Tests one fold of 262144 elements, which takes the cooperative shape.
         * @see tests/unit/README.md
         */
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

            SushiBLAS::ReductionOps(*engine).mean_spatial(x, out);
            engine->execute().value();

            double acc = 0.0;
            for (std::size_t i = 0; i < total; ++i)
                acc += static_cast<double>(src[i]);
            const double expected = acc / static_cast<double>(total);

            const std::vector<float> got = read_back(out);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(1));
            EXPECT_NEAR(got[0], static_cast<float>(expected), 1e-5f);
        }

        /**
         * @brief Tests that a span with no second segment axis reduces as it always did.
         * @see tests/unit/README.md
         */
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

            SushiBLAS::ReductionOps(*engine).sum_along(in, 1, out);
            engine->execute().value();

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
