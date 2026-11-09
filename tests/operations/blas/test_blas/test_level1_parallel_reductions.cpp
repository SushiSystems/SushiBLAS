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
 * @file test_level1_parallel_reductions.cpp
 * @brief Tests the Level 1 reductions on the deterministic machinery and HALF accumulation.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /**
             * @brief Holds the rounded operations nrm2 may add to the sum of squares: the square root
             *        and the rescaling, four ulps in all, which is eight unit roundoffs.
             */
            constexpr std::size_t K_NRM2_ROUNDINGS = 8;

            /** @brief Holds the ratio nrm2 is held to: the bound above is the whole allowance. */
            constexpr double K_NRM2_THRESHOLD = 1.0;

            /** @brief Holds the length of the vectors the determinism and tie tests use. */
            constexpr int64_t K_LONG_LENGTH = 100003;

            /** @brief Holds the length of the vector of the tie test: long enough for two stages on any device. */
            constexpr std::size_t K_TIE_LENGTH = 1u << 20;

            /** @brief Holds the first index of the run of equal magnitudes in the tie test. */
            constexpr std::size_t K_TIE_START = 12345;

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

            /** @brief Returns the vector whose element i is element(i), n elements long. */
            template<typename Element>
            [[nodiscard]] std::vector<float> vector_of(int64_t n, Element element)
            {
                std::vector<float> v(static_cast<std::size_t>(n));
                for (std::size_t i = 0; i < v.size(); ++i)
                    v[i] = element(i);
                return v;
            }

            /** @brief Returns x_at(0..n). */
            [[nodiscard]] std::vector<float> x_vector(int64_t n)
            {
                return vector_of(n, x_at);
            }

            /** @brief Returns y_at(0..n). */
            [[nodiscard]] std::vector<float> y_vector(int64_t n)
            {
                return vector_of(n, y_at);
            }

            /** @brief Returns a x operand whose element at two thirds of the way is the unique peak, -2. */
            [[nodiscard]] std::vector<float> peaked_vector(int64_t n)
            {
                std::vector<float> v = x_vector(n);
                v[static_cast<std::size_t>(n - 1) * 2 / 3] = -2.0f;
                return v;
            }

            /** @brief Returns the index a serial left-to-right IAMAX scan picks. */
            [[nodiscard]] int64_t serial_iamax(const std::vector<float>& v)
            {
                int64_t best = 0;
                for (std::size_t i = 1; i < v.size(); ++i)
                    if (std::fabs(v[i]) > std::fabs(v[static_cast<std::size_t>(best)]))
                        best = static_cast<int64_t>(i);
                return best;
            }

            /** @brief Returns the sum of x_at(i) * y_at(i) in double precision. */
            [[nodiscard]] double host_dot(const std::vector<float>& x, const std::vector<float>& y)
            {
                double sum = 0.0;
                for (std::size_t i = 0; i < x.size(); ++i)
                    sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
                return sum;
            }

            /** @brief Returns the sum of absolute values in double precision. */
            [[nodiscard]] double host_asum(const std::vector<float>& x)
            {
                double sum = 0.0;
                for (const float v : x)
                    sum += std::fabs(static_cast<double>(v));
                return sum;
            }

            /** @brief Returns the Euclidean norm in double precision. */
            [[nodiscard]] double host_nrm2(const std::vector<float>& x)
            {
                return std::sqrt(host_dot(x, x));
            }
        } // namespace

        /**
         * @class Level1SyclEngine
         * @brief Starts the engine with the SYCL kernels forced, so the reference path cannot answer instead.
         */
        class Level1SyclEngine : public SushiBLASTest
        {
            protected:
                /** @brief Forces the SYCL kernels. */
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
        };

        /**
         * @class Medium_Level1ReductionLengths
         * @brief Runs one SYCL Level 1 reduction per vector length against a host reference.
         */
        class Medium_Level1ReductionLengths : public Level1SyclEngine,
                                              public ::testing::WithParamInterface<int64_t>
        {
        };

        /** @brief Returns the name of a length row: Length<n>. */
        [[nodiscard]] std::string length_name(const ::testing::TestParamInfo<int64_t>& info)
        {
            return "Length" + std::to_string(info.param);
        }

        /**
         * @brief Tests dot against a double-precision host sum, compared for equality.
         *
         * Every product is a multiple of 1/32 and every partial sum stays below 2^17, where FLOAT32
         * holds each multiple of 1/32 exactly, so any combination order gives the exact sum.
         * @see tests/operations/blas/README.md
         */
        TEST_P(Medium_Level1ReductionLengths, DotMatchesTheHostSum)
        {
            const auto hx = x_vector(GetParam());
            const auto hy = y_vector(GetParam());

            auto x = upload(hx);
            auto y = upload(hy);
            auto r = engine->create_tensor({1});
            SushiBLAS::BLASOps(recorder()).dot(x, y, r);
            recording->run().value();

            EXPECT_EQ(static_cast<double>(read_scalar<float>(r)), host_dot(hx, hy));
        }

        /**
         * @brief Tests asum against a double-precision host sum, compared for equality as dot is.
         * @see tests/operations/blas/README.md
         */
        TEST_P(Medium_Level1ReductionLengths, AsumMatchesTheHostSum)
        {
            const auto hx = x_vector(GetParam());

            auto x = upload(hx);
            auto r = engine->create_tensor({1});
            SushiBLAS::BLASOps(recorder()).asum(x, r);
            recording->run().value();

            EXPECT_EQ(static_cast<double>(read_scalar<float>(r)), host_asum(hx));
        }

        /**
         * @brief Tests nrm2 against a double-precision host norm within eight unit roundoffs.
         * @see tests/operations/blas/README.md
         */
        TEST_P(Medium_Level1ReductionLengths, Nrm2MatchesTheHostNorm)
        {
            const auto hx = x_vector(GetParam());

            auto x = upload(hx);
            auto r = engine->create_tensor({1});
            SushiBLAS::BLASOps(recorder()).nrm2(x, r);
            recording->run().value();

            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
                static_cast<double>(read_scalar<float>(r)), host_nrm2(hx), K_NRM2_ROUNDINGS,
                K_NRM2_THRESHOLD)));
        }

        /**
         * @brief Tests iamax against a serial host scan over a vector with a unique late peak.
         * @see tests/operations/blas/README.md
         */
        TEST_P(Medium_Level1ReductionLengths, IamaxMatchesTheSerialScan)
        {
            const auto hx = peaked_vector(GetParam());

            auto x = upload(hx);
            auto r = engine->create_tensor({1}, Core::DataType::Int64);
            SushiBLAS::BLASOps(recorder()).iamax(x, r);
            recording->run().value();

            EXPECT_EQ(read_scalar<int64_t>(r), serial_iamax(hx));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_Level1ReductionLengths,
                                 ::testing::Values<int64_t>(1, 7, 1000, K_LONG_LENGTH),
                                 length_name);

        /**
         * @class Medium_Level1ParallelReductions
         * @brief Runs the SYCL Level 1 reductions whose promise is not a value at a length.
         */
        class Medium_Level1ParallelReductions : public Level1SyclEngine
        {
        };

        /**
         * @brief Tests that two dot products of the same inputs return identical bits.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, DotRepeatsToTheBit)
        {
            const auto hx = vector_of(K_LONG_LENGTH, [](std::size_t i)
            {
                return 0.1f + static_cast<float>(i % 97) * 0.0137f;
            });
            const auto hy = vector_of(K_LONG_LENGTH, [](std::size_t i)
            {
                return 1.3f - static_cast<float>(i % 89) * 0.0291f;
            });

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
         * @brief Tests that iamax picks the lowest index of a run of equal magnitudes that spans many work-groups.
         *
         * Every element from K_TIE_START on has magnitude 9, so whichever way the device splits the
         * vector the tie crosses work-groups, and the answer is K_TIE_START.
         * @see tests/operations/blas/README.md
         */
        TEST_F(Medium_Level1ParallelReductions, IamaxTieAcrossWorkGroupsTakesTheLowerIndex)
        {
            std::vector<float> hx(K_TIE_LENGTH, 0.5f);
            for (std::size_t i = K_TIE_START; i < hx.size(); ++i)
                hx[i] = (i % 2 == 0) ? 9.0f : -9.0f;

            auto x = upload(hx);
            auto r = engine->create_tensor({1}, Core::DataType::Int64);
            SushiBLAS::BLASOps(recorder()).iamax(x, r);
            recording->run().value();

            EXPECT_EQ(read_scalar<int64_t>(r), static_cast<int64_t>(K_TIE_START));
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
