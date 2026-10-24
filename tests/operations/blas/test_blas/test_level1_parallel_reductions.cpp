/**************************************************************************/
/* test_level1_parallel_reductions.cpp                                    */
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
 * @file test_level1_parallel_reductions.cpp
 * @brief Tests the Level 1 reductions on the deterministic machinery and HALF accumulation.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The lengths every Level 1 reduction is checked at. */
            const std::vector<int64_t> K_LENGTHS = {1, 7, 1000, 100003};

            /**
             * @brief Allowed difference for dot and asum: none.
             *
             * Every x and y element is a multiple of 1/8 and 1/4 with magnitude at most 0.75 and
             * 1.25, so each product and magnitude is a multiple of 1/32. At n = 100003 every
             * partial sum stays below 100003 * 0.9375 < 2^17, where FLOAT32 holds every multiple
             * of 1/32 exactly, so any combination order gives the exact sum.
             */
            constexpr double K_EXACT_TOLERANCE = 0.0;

            /** @brief Relative tolerance for nrm2: four FLOAT32 ulps for the square root and rescaling. */
            constexpr double K_NRM2_RELATIVE_TOLERANCE = 4.0 * 1.1920928955078125e-7;

            /** @brief Returns element i of the x operand: ((7i mod 13) - 6) / 8. */
            [[nodiscard]] float x_at(std::size_t i)
            {
                return static_cast<float>(static_cast<int>((i * 7) % 13) - 6) * 0.125f;
            }

            /** @brief Returns element i of the y operand: ((5i mod 11) - 5) / 4. */
            [[nodiscard]] float y_at(std::size_t i)
            {
                return static_cast<float>(static_cast<int>((i * 5) % 11) - 5) * 0.25f;
            }
        } // namespace

        /**
         * @class Medium_Level1ParallelReductions
         * @brief Runs the SYCL Level 1 reductions against host-computed references.
         */
        class Medium_Level1ParallelReductions : public SushiBLASTest
        {
            protected:
                /** @brief Forces the SYCL kernels so the reference path cannot answer instead. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    set_mode(ExecutionMode::ForceSYCL);
                }

                /** @brief Returns a FLOAT32 tensor holding @p host. */
                Tensor upload(const std::vector<float>& host)
                {
                    auto t = engine->create_tensor({static_cast<int64_t>(host.size())});
                    TestEnvironment::queue()
                        .memcpy(t.data(), host.data(), host.size() * sizeof(float)).wait();
                    return t;
                }

                /** @brief Returns the single element of @p t as a T. */
                template<typename T>
                [[nodiscard]] T read_scalar(const Tensor& t)
                {
                    T value{};
                    TestEnvironment::queue()
                        .memcpy(&value, t.data(), sizeof(T)).wait();
                    return value;
                }

                /** @brief Returns x_at(0..n). */
                [[nodiscard]] static std::vector<float> x_vector(int64_t n)
                {
                    std::vector<float> v(static_cast<std::size_t>(n));
                    for (std::size_t i = 0; i < v.size(); ++i)
                        v[i] = x_at(i);
                    return v;
                }

                /** @brief Returns y_at(0..n). */
                [[nodiscard]] static std::vector<float> y_vector(int64_t n)
                {
                    std::vector<float> v(static_cast<std::size_t>(n));
                    for (std::size_t i = 0; i < v.size(); ++i)
                        v[i] = y_at(i);
                    return v;
                }

                /** @brief Returns the index a serial left-to-right IAMAX scan picks. */
                [[nodiscard]] static int64_t serial_iamax(const std::vector<float>& v)
                {
                    int64_t best = 0;
                    for (std::size_t i = 1; i < v.size(); ++i)
                        if (std::fabs(v[i]) > std::fabs(v[static_cast<std::size_t>(best)]))
                            best = static_cast<int64_t>(i);
                    return best;
                }
        };

        /**
         * @brief Tests dot against a double-precision host sum at every length.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, DotMatchesTheHostSum)
        {
            for (int64_t n : K_LENGTHS)
            {
                SCOPED_TRACE(n);
                const auto hx = x_vector(n);
                const auto hy = y_vector(n);

                double expected = 0.0;
                for (std::size_t i = 0; i < hx.size(); ++i)
                    expected += static_cast<double>(hx[i]) * static_cast<double>(hy[i]);

                auto x = upload(hx);
                auto y = upload(hy);
                auto r = engine->create_tensor({1});
                SushiBLAS::BLASOps(recorder()).dot(x, y, r);
                recording->run().value();

                EXPECT_NEAR(static_cast<double>(read_scalar<float>(r)), expected, K_EXACT_TOLERANCE);
            }
        }

        /**
         * @brief Tests asum against a double-precision host sum at every length.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, AsumMatchesTheHostSum)
        {
            for (int64_t n : K_LENGTHS)
            {
                SCOPED_TRACE(n);
                const auto hx = x_vector(n);

                double expected = 0.0;
                for (float v : hx)
                    expected += std::fabs(static_cast<double>(v));

                auto x = upload(hx);
                auto r = engine->create_tensor({1});
                SushiBLAS::BLASOps(recorder()).asum(x, r);
                recording->run().value();

                EXPECT_NEAR(static_cast<double>(read_scalar<float>(r)), expected, K_EXACT_TOLERANCE);
            }
        }

        /**
         * @brief Tests nrm2 against a double-precision host norm at every length.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, Nrm2MatchesTheHostNorm)
        {
            for (int64_t n : K_LENGTHS)
            {
                SCOPED_TRACE(n);
                const auto hx = x_vector(n);

                double ssq = 0.0;
                for (float v : hx)
                    ssq += static_cast<double>(v) * static_cast<double>(v);
                const double expected = std::sqrt(ssq);

                auto x = upload(hx);
                auto r = engine->create_tensor({1});
                SushiBLAS::BLASOps(recorder()).nrm2(x, r);
                recording->run().value();

                EXPECT_NEAR(static_cast<double>(read_scalar<float>(r)), expected,
                            expected * K_NRM2_RELATIVE_TOLERANCE);
            }
        }

        /**
         * @brief Tests iamax against a serial host scan at every length.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, IamaxMatchesTheSerialScan)
        {
            for (int64_t n : K_LENGTHS)
            {
                SCOPED_TRACE(n);
                auto hx = x_vector(n);
                // A unique peak late in the vector, so the answer is not index 0.
                hx[static_cast<std::size_t>(n - 1) * 2 / 3] = -2.0f;
                const int64_t expected = serial_iamax(hx);

                auto x = upload(hx);
                auto r = engine->create_tensor({1}, Core::DataType::Int64);
                SushiBLAS::BLASOps(recorder()).iamax(x, r);
                recording->run().value();

                EXPECT_EQ(read_scalar<int64_t>(r), expected);
            }
        }

        /**
         * @brief Tests that two dot products of the same inputs return identical bits.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, DotRepeatsToTheBit)
        {
            constexpr int64_t K_N = 100003;

            std::vector<float> hx(static_cast<std::size_t>(K_N));
            std::vector<float> hy(static_cast<std::size_t>(K_N));
            for (std::size_t i = 0; i < hx.size(); ++i)
            {
                hx[i] = 0.1f + static_cast<float>(i % 97) * 0.0137f;
                hy[i] = 1.3f - static_cast<float>(i % 89) * 0.0291f;
            }

            auto x = upload(hx);
            auto y = upload(hy);
            auto first = engine->create_tensor({1});
            auto second = engine->create_tensor({1});

            SushiBLAS::BLASOps(recorder()).dot(x, y, first);
            recording->run().value();
            SushiBLAS::BLASOps(recorder()).dot(x, y, second);
            recording->run().value();

            const float a = read_scalar<float>(first);
            const float b = read_scalar<float>(second);
            EXPECT_EQ(std::memcmp(&a, &b, sizeof(float)), 0);
        }

        /**
         * @brief Tests that iamax picks the lower index of a tie whose halves sit in different work-groups.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, IamaxTieAcrossWorkGroupsTakesTheLowerIndex)
        {
            auto& q = TestEnvironment::queue();
            const Device::Profile& profile = Device::profile_for(q);

            // Long enough that the two-stage shape takes it on any device this suite runs on.
            constexpr std::size_t K_N = 1u << 20;
            const Reduce::Plan geometry = Reduce::plan(profile, K_N, Reduce::K_MAX_GROUPS);
            const std::size_t local = Reduce::group_width(profile);
            const std::size_t stride = (geometry.groups < 2 ? 2 : geometry.groups) * local;

            // Index stride + local is read by work-item local (group 1); index 2 * stride by work-item 0 (group 0).
            const std::size_t lower = stride + local;
            const std::size_t higher = 2 * stride;
            ASSERT_LT(higher, K_N);

            std::vector<float> hx(K_N, 0.5f);
            hx[lower] = -9.0f;
            hx[higher] = 9.0f;

            auto x = upload(hx);
            auto r = engine->create_tensor({1}, Core::DataType::Int64);
            SushiBLAS::BLASOps(recorder()).iamax(x, r);
            recording->run().value();

            EXPECT_EQ(read_scalar<int64_t>(r), static_cast<int64_t>(lower));
        }

        /**
         * @brief Tests that a HALF mean accumulates in FLOAT32, so a sum past 65504 does not overflow on the way.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, HalfMeanAccumulatesInFloat)
        {
            // 60000 is a multiple of 32, the HALF spacing between 32768 and 65504, so it is exact.
            const float K_HALF_ELEMENT = 60000.0f;
            const float K_EXPECTED_MEAN = 60000.0f;

            const std::vector<sycl::half> host = {sycl::half(K_HALF_ELEMENT), sycl::half(K_HALF_ELEMENT)};
            auto t = create_initialized_tensor<sycl::half>({2}, host);
            auto r = engine->create_tensor({1}, Core::DataType::Half);

            SushiBLAS::ReductionOps(recorder()).mean(t, r);
            recording->run().value();

            EXPECT_EQ(static_cast<float>(read_scalar<sycl::half>(r)), K_EXPECTED_MEAN);
        }

        /**
         * @brief Tests that a HALF sum of 60000 and 60000 rounds once, to infinity, since 120000 exceeds HALF.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, HalfSumRoundsOnceIntoTheHalfResult)
        {
            const float K_HALF_ELEMENT = 60000.0f;

            const std::vector<sycl::half> host = {sycl::half(K_HALF_ELEMENT), sycl::half(K_HALF_ELEMENT)};
            auto t = create_initialized_tensor<sycl::half>({2}, host);
            auto r = engine->create_tensor({1}, Core::DataType::Half);

            SushiBLAS::ReductionOps(recorder()).sum(t, r);
            recording->run().value();

            const float value = static_cast<float>(read_scalar<sycl::half>(r));
            EXPECT_TRUE(std::isinf(value) && value > 0.0f);
        }

    } // namespace Test
} // namespace SushiBLAS
