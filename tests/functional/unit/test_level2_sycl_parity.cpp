/**************************************************************************/
/* test_level2_sycl_parity.cpp                                            */
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
 * @file test_level2_sycl_parity.cpp
 * @brief Tests that the SYCL GEMV and SYMV kernels match the reference kernels and repeat to the bit.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute error allowed per reduced term, for float.
        constexpr double K_FLOAT_ERROR_PER_TERM = 1e-6;

        /// @brief Absolute error allowed per reduced term, for double.
        constexpr double K_DOUBLE_ERROR_PER_TERM = 1e-14;

        /// @brief Relative error allowed against the reference value, for every type.
        constexpr double K_RELATIVE_ERROR = 1e-5;

        /// @brief Edge of the small square case; below every work-group-per-row crossover.
        constexpr int64_t K_SMALL_N = 100;

        /// @brief Edge of the large square case.
        constexpr int64_t K_LARGE_N = 4096;

        /// @brief Rows of the wide GEMV case.
        constexpr int64_t K_WIDE_ROWS = 4;

        /// @brief Columns of the wide GEMV case; at the CPU two-stage reduction threshold.
        constexpr int64_t K_WIDE_COLS = 16384;

        /// @brief Scales op(A) * x in every call of this file.
        constexpr double K_ALPHA = 1.5;

        /// @brief Scales the prior y in every call of this file.
        constexpr double K_BETA = -0.5;

        /**
         * @brief Returns a fixed, sign-mixed value in [-1, 1] for index i.
         */
        [[nodiscard]] inline double pattern_value(std::size_t i) noexcept
        {
            return static_cast<double>(static_cast<int64_t>((i * 37u) % 101u) - 50) / 50.0;
        }

        /**
         * @brief Returns the per-term error allowance of T.
         */
        template<typename T>
        [[nodiscard]] constexpr double error_per_term() noexcept
        {
            return std::is_same<T, float>::value ? K_FLOAT_ERROR_PER_TERM : K_DOUBLE_ERROR_PER_TERM;
        }

        /**
         * @brief Runs GEMV and SYMV in a chosen execution mode and reads the results back.
         */
        template<typename T>
        class Unit_Level2SyclParityTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a tensor of dims filled with pattern_value, offset by seed. */
                [[nodiscard]] Tensor patterned(std::initializer_list<int64_t> dims, std::size_t seed)
                {
                    Tensor t = engine->create_tensor(dims, get_sushiblas_dtype<T>());
                    std::vector<T> host(static_cast<std::size_t>(t.num_elements));
                    for (std::size_t i = 0; i < host.size(); ++i)
                        host[i] = static_cast<T>(pattern_value(i + seed));
                    TestEnvironment::get_context().get_queue()
                        .memcpy(t.data(), host.data(), host.size() * sizeof(T)).wait();
                    return t;
                }

                /** @brief Copies a tensor to the host. */
                [[nodiscard]] static std::vector<T> read_back(const Tensor& t)
                {
                    std::vector<T> host(static_cast<std::size_t>(t.num_elements));
                    TestEnvironment::get_context().get_queue()
                        .memcpy(host.data(), t.data(), host.size() * sizeof(T)).wait();
                    return host;
                }

                /** @brief Returns y = alpha * A * x + beta * y for an m x n A, computed in mode. */
                [[nodiscard]] std::vector<T> run_gemv(ExecutionMode mode, int64_t m, int64_t n)
                {
                    engine->set_mode(mode);
                    Tensor A = patterned({m, n}, 0);
                    Tensor x = patterned({n}, 11);
                    Tensor y = patterned({m}, 23);
                    engine->blas().gemv(A, x, y, false, K_ALPHA, K_BETA);
                    engine->execute().wait();
                    return read_back(y);
                }

                /** @brief Returns y = alpha * A * x + beta * y for a symmetric n x n A, computed in mode. */
                [[nodiscard]] std::vector<T> run_symv(ExecutionMode mode, int64_t n, bool upper)
                {
                    engine->set_mode(mode);
                    Tensor A = patterned({n, n}, 0);
                    Tensor x = patterned({n}, 11);
                    Tensor y = patterned({n}, 23);
                    engine->blas().symv(A, x, y, upper, K_ALPHA, K_BETA);
                    engine->execute().wait();
                    return read_back(y);
                }

                /** @brief Expects every element within the allowance of a reduction over len terms. */
                static void expect_close(const std::vector<T>& actual, const std::vector<T>& expected,
                                         int64_t len)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    const double absolute = error_per_term<T>() * static_cast<double>(len);
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const double e = static_cast<double>(expected[i]);
                        const double bound = absolute + K_RELATIVE_ERROR * std::fabs(e);
                        EXPECT_NEAR(static_cast<double>(actual[i]), e, bound) << "index " << i;
                    }
                }

                /** @brief Expects two results to hold the same bits. */
                static void expect_same_bits(const std::vector<T>& first,
                                             const std::vector<T>& second)
                {
                    ASSERT_EQ(first.size(), second.size());
                    EXPECT_EQ(0, std::memcmp(first.data(), second.data(), first.size() * sizeof(T)));
                }
        };

        /// @brief Element types the parity tests run on.
        using ParityTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_Level2SyclParityTest, ParityTypes);

        TYPED_TEST(Unit_Level2SyclParityTest, GemvMatchesReferenceSmall)
        {
            const auto expected = this->run_gemv(ExecutionMode::FORCE_REFERENCE, K_SMALL_N, K_SMALL_N);
            const auto actual = this->run_gemv(ExecutionMode::FORCE_SYCL, K_SMALL_N, K_SMALL_N);
            this->expect_close(actual, expected, K_SMALL_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvMatchesReferenceLarge)
        {
            const auto expected = this->run_gemv(ExecutionMode::FORCE_REFERENCE, K_LARGE_N, K_LARGE_N);
            const auto actual = this->run_gemv(ExecutionMode::FORCE_SYCL, K_LARGE_N, K_LARGE_N);
            this->expect_close(actual, expected, K_LARGE_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvMatchesReferenceWide)
        {
            const auto expected = this->run_gemv(ExecutionMode::FORCE_REFERENCE, K_WIDE_ROWS, K_WIDE_COLS);
            const auto actual = this->run_gemv(ExecutionMode::FORCE_SYCL, K_WIDE_ROWS, K_WIDE_COLS);
            this->expect_close(actual, expected, K_WIDE_COLS);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, SymvMatchesReferenceSmall)
        {
            for (const bool upper : {true, false})
            {
                const auto expected = this->run_symv(ExecutionMode::FORCE_REFERENCE, K_SMALL_N, upper);
                const auto actual = this->run_symv(ExecutionMode::FORCE_SYCL, K_SMALL_N, upper);
                this->expect_close(actual, expected, K_SMALL_N);
            }
        }

        TYPED_TEST(Unit_Level2SyclParityTest, SymvMatchesReferenceLarge)
        {
            for (const bool upper : {true, false})
            {
                const auto expected = this->run_symv(ExecutionMode::FORCE_REFERENCE, K_LARGE_N, upper);
                const auto actual = this->run_symv(ExecutionMode::FORCE_SYCL, K_LARGE_N, upper);
                this->expect_close(actual, expected, K_LARGE_N);
            }
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvRepeatsToTheBit)
        {
            const auto first = this->run_gemv(ExecutionMode::FORCE_SYCL, K_WIDE_ROWS, K_WIDE_COLS);
            const auto second = this->run_gemv(ExecutionMode::FORCE_SYCL, K_WIDE_ROWS, K_WIDE_COLS);
            this->expect_same_bits(first, second);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, SymvRepeatsToTheBit)
        {
            const auto first = this->run_symv(ExecutionMode::FORCE_SYCL, K_LARGE_N, true);
            const auto second = this->run_symv(ExecutionMode::FORCE_SYCL, K_LARGE_N, true);
            this->expect_same_bits(first, second);
        }
    } // namespace Test
} // namespace SushiBLAS
