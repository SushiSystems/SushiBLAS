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
 * @brief Tests that the GEMV and SYMV kernels stay within their rounding bound of a
 *        double-precision truth, and that the SYCL kernels repeat to the bit.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>
#include "test_utils.hpp"
#include "blas_flags.hpp"
#include "blas_truth.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Edge of the small square case; below every work-group-per-row crossover.
        constexpr int64_t K_SMALL_N = 100;

        /// @brief Edge of the large square case.
        constexpr int64_t K_LARGE_N = 4096;

        /// @brief Rows of the wide GEMV case.
        constexpr int64_t K_WIDE_ROWS = 4;

        /// @brief Columns of the wide GEMV case; at the CPU two-stage reduction threshold.
        constexpr int64_t K_WIDE_COLS = 16384;

        /// @brief Index of a data set at which the elements of A start.
        constexpr std::size_t K_SEED_A = 0;

        /// @brief Index of a data set at which the elements of x start.
        constexpr std::size_t K_SEED_X = 11;

        /// @brief Index of a data set at which the elements of y start.
        constexpr std::size_t K_SEED_Y = 23;

        /// @brief Roundings a kernel makes whatever the length, and two of the truth itself.
        constexpr std::size_t K_FIXED_ROUNDINGS = 6;

        /// @brief Roundings counted for a compensated sum, which does not grow with the length.
        constexpr std::size_t K_COMPENSATED_ROUNDINGS = 2;

        /**
         * @brief Returns a fixed, sign-mixed value in [-1, 1] for index i.
         */
        [[nodiscard]] inline double pattern_value(std::size_t i) noexcept
        {
            return static_cast<double>(static_cast<int64_t>((i * 37u) % 101u) - 50) / 50.0;
        }

        /**
         * @brief Returns a fixed integer in [-3, 3] for index i.
         */
        [[nodiscard]] inline double integer_value(std::size_t i) noexcept
        {
            return static_cast<double>(static_cast<int64_t>((i * 37u) % 7u) - 3);
        }

        /** @brief Holds the values a case fills its operands with and the scalars of its call. */
        struct ParityData
        {
            double (*value)(std::size_t) noexcept; /**< The value at an index. */
            double alpha;                          /**< Scales op(A) * x. */
            double beta;                           /**< Scales the prior y. */
        };

        /// @brief Rounded data: sign-mixed fractions, alpha 1.5 and beta -0.5.
        constexpr ParityData K_PATTERNED{&pattern_value, 1.5, -0.5};

        /// @brief Exact data: every partial sum of every case is an integer below 2^24.
        constexpr ParityData K_INTEGER_VALUED{&integer_value, 2.0, -1.0};

        /**
         * @brief Runs GEMV and SYMV in a chosen execution mode and measures them against a truth.
         * @see tests/common/README.md
         */
        template<typename T>
        class Unit_Level2SyclParityTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns the element a tensor of T holds at index of data, as a double. */
                [[nodiscard]] static double element(const ParityData& data,
                                                    std::size_t index) noexcept
                {
                    return static_cast<double>(static_cast<T>(data.value(index)));
                }

                /** @brief Returns a tensor of dims filled from data, starting at index seed. */
                [[nodiscard]] Tensor filled(std::initializer_list<int64_t> dims,
                                            const ParityData& data, std::size_t seed)
                {
                    Tensor t = engine->create_tensor(dims, get_sushiblas_dtype<T>());
                    std::vector<T> host(static_cast<std::size_t>(t.num_elements));
                    for (std::size_t i = 0; i < host.size(); ++i)
                        host[i] = static_cast<T>(data.value(i + seed));
                    TestEnvironment::get_context().devices().queue(0)
                        .memcpy(t.data(), host.data(), host.size() * sizeof(T)).wait();
                    return t;
                }

                /** @brief Copies a tensor to the host. */
                [[nodiscard]] static std::vector<T> read_back(const Tensor& t)
                {
                    std::vector<T> host(static_cast<std::size_t>(t.num_elements));
                    TestEnvironment::get_context().devices().queue(0)
                        .memcpy(host.data(), t.data(), host.size() * sizeof(T)).wait();
                    return host;
                }

                /** @brief Returns y = alpha * A * x + beta * y for an m x n A, computed in mode. */
                [[nodiscard]] std::vector<T> run_gemv(ExecutionMode mode, int64_t m, int64_t n,
                                                      const ParityData& data = K_PATTERNED)
                {
                    engine->set_mode(mode);
                    Tensor A = filled({m, n}, data, K_SEED_A);
                    Tensor x = filled({n}, data, K_SEED_X);
                    Tensor y = filled({m}, data, K_SEED_Y);
                    SushiBLAS::BLASOps(*engine).gemv(A, x, y, Core::Transpose::NoTrans,
                                                     data.alpha, data.beta);
                    engine->execute();
                    return read_back(y);
                }

                /** @brief Returns y = alpha * A * x + beta * y for a symmetric A, run in mode. */
                [[nodiscard]] std::vector<T> run_symv(ExecutionMode mode, int64_t n, bool upper,
                                                      const ParityData& data = K_PATTERNED)
                {
                    engine->set_mode(mode);
                    Tensor A = filled({n, n}, data, K_SEED_A);
                    Tensor x = filled({n}, data, K_SEED_X);
                    Tensor y = filled({n}, data, K_SEED_Y);
                    SushiBLAS::BLASOps(*engine).symv(A, x, y, uplo_of(upper), data.alpha,
                                                     data.beta);
                    engine->execute();
                    return read_back(y);
                }

                /** @brief Returns the exact result of run_gemv on data, element by element. */
                [[nodiscard]] static std::vector<TruthValue> truth_gemv(int64_t m, int64_t n,
                                                                        const ParityData& data)
                {
                    const std::size_t rows = static_cast<std::size_t>(m);
                    const std::size_t cols = static_cast<std::size_t>(n);
                    std::vector<TruthValue> truth(rows);
                    for (std::size_t i = 0; i < rows; ++i)
                    {
                        TruthSum sum;
                        for (std::size_t j = 0; j < cols; ++j)
                            sum.add_product(data.alpha * element(data, K_SEED_A + i * cols + j),
                                            element(data, K_SEED_X + j));
                        sum.add_product(data.beta, element(data, K_SEED_Y + i));
                        truth[i] = sum.value();
                    }
                    return truth;
                }

                /** @brief Returns the exact result of run_symv on data, read from one triangle. */
                [[nodiscard]] static std::vector<TruthValue> truth_symv(int64_t n, bool upper,
                                                                        const ParityData& data)
                {
                    const std::size_t order = static_cast<std::size_t>(n);
                    std::vector<TruthValue> truth(order);
                    for (std::size_t i = 0; i < order; ++i)
                    {
                        TruthSum sum;
                        for (std::size_t j = 0; j < order; ++j)
                        {
                            const bool stored = upper ? i <= j : i >= j;
                            const std::size_t index = stored ? i * order + j : j * order + i;
                            sum.add_product(data.alpha * element(data, K_SEED_A + index),
                                            element(data, K_SEED_X + j));
                        }
                        sum.add_product(data.beta, element(data, K_SEED_Y + i));
                        truth[i] = sum.value();
                    }
                    return truth;
                }

                /** @brief Returns the unit roundoff of the type the kernel of mode sums T in. */
                [[nodiscard]] static double summing_unit(ExecutionMode mode) noexcept
                {
                    return mode == ExecutionMode::ForceReference
                               ? unit_roundoff<Kernels::ReferenceAccumulator<T>>()
                               : unit_roundoff<Kernels::Accumulator<T>>();
                }

                /** @brief Returns the roundings the kernel of mode makes on a sum of len terms. */
                [[nodiscard]] static std::size_t roundings(ExecutionMode mode,
                                                           int64_t len) noexcept
                {
                    const bool compensated = mode == ExecutionMode::ForceReference &&
                                             Kernels::ReferenceSum<T>::K_IS_COMPENSATED;
                    return (compensated ? K_COMPENSATED_ROUNDINGS : static_cast<std::size_t>(len)) +
                           K_FIXED_ROUNDINGS;
                }

                /** @brief Expects every element within the bound of mode for a sum of len terms. */
                static void expect_within_bound(const std::vector<T>& actual,
                                                const std::vector<TruthValue>& truth, int64_t len,
                                                ExecutionMode mode)
                {
                    ASSERT_EQ(actual.size(), truth.size());
                    const double unit = summing_unit(mode);
                    const std::size_t count = roundings(mode, len);
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const double bound = rounding_bound(count, unit, truth[i].magnitude) +
                                             unit_roundoff<T>() * std::fabs(truth[i].value);
                        EXPECT_NEAR(static_cast<double>(actual[i]), truth[i].value, bound)
                            << "index " << i;
                    }
                }

                /** @brief Expects every element to equal the truth with no tolerance. */
                static void expect_exact(const std::vector<T>& actual,
                                         const std::vector<TruthValue>& truth)
                {
                    ASSERT_EQ(actual.size(), truth.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        EXPECT_EQ(static_cast<double>(actual[i]), truth[i].value) << "index " << i;
                }

                /** @brief Expects two results to hold the same bits. */
                static void expect_same_bits(const std::vector<T>& first,
                                             const std::vector<T>& second)
                {
                    ASSERT_EQ(first.size(), second.size());
                    EXPECT_EQ(0, std::memcmp(first.data(), second.data(), first.size() * sizeof(T)));
                }

                /** @brief Expects GEMV of an m x n A within its bound of the truth in each mode. */
                void expect_gemv_within_bound(int64_t m, int64_t n)
                {
                    const auto truth = truth_gemv(m, n, K_PATTERNED);
                    for (const ExecutionMode mode : K_MODES)
                        expect_within_bound(run_gemv(mode, m, n), truth, n, mode);
                }

                /** @brief Expects SYMV of order n within its bound, for each mode and triangle. */
                void expect_symv_within_bound(int64_t n)
                {
                    for (const bool upper : {true, false})
                    {
                        const auto truth = truth_symv(n, upper, K_PATTERNED);
                        for (const ExecutionMode mode : K_MODES)
                            expect_within_bound(run_symv(mode, n, upper), truth, n, mode);
                    }
                }

                /** @brief Expects GEMV of an m x n A of small integers to equal the truth. */
                void expect_gemv_exact(int64_t m, int64_t n)
                {
                    const auto truth = truth_gemv(m, n, K_INTEGER_VALUED);
                    for (const ExecutionMode mode : K_MODES)
                        expect_exact(run_gemv(mode, m, n, K_INTEGER_VALUED), truth);
                }

                /** @brief Expects SYMV of order n of small integers to equal the truth. */
                void expect_symv_exact(int64_t n)
                {
                    for (const bool upper : {true, false})
                    {
                        const auto truth = truth_symv(n, upper, K_INTEGER_VALUED);
                        for (const ExecutionMode mode : K_MODES)
                            expect_exact(run_symv(mode, n, upper, K_INTEGER_VALUED), truth);
                    }
                }

                /// @brief The two modes every truth case runs in.
                static constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                            ExecutionMode::ForceSycl};
        };

        /// @brief Element types the parity tests run on.
        using ParityTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_Level2SyclParityTest, ParityTypes);

        TYPED_TEST(Unit_Level2SyclParityTest, GemvIsWithinItsBoundOfTheTruthSmall)
        {
            this->expect_gemv_within_bound(K_SMALL_N, K_SMALL_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvIsWithinItsBoundOfTheTruthLarge)
        {
            this->expect_gemv_within_bound(K_LARGE_N, K_LARGE_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvIsWithinItsBoundOfTheTruthWide)
        {
            this->expect_gemv_within_bound(K_WIDE_ROWS, K_WIDE_COLS);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, SymvIsWithinItsBoundOfTheTruthSmall)
        {
            this->expect_symv_within_bound(K_SMALL_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, SymvIsWithinItsBoundOfTheTruthLarge)
        {
            this->expect_symv_within_bound(K_LARGE_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvIsExactOnSmallIntegersLarge)
        {
            this->expect_gemv_exact(K_LARGE_N, K_LARGE_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvIsExactOnSmallIntegersWide)
        {
            this->expect_gemv_exact(K_WIDE_ROWS, K_WIDE_COLS);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, SymvIsExactOnSmallIntegersLarge)
        {
            this->expect_symv_exact(K_LARGE_N);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, GemvRepeatsToTheBit)
        {
            const auto first = this->run_gemv(ExecutionMode::ForceSycl, K_WIDE_ROWS, K_WIDE_COLS);
            const auto second = this->run_gemv(ExecutionMode::ForceSycl, K_WIDE_ROWS, K_WIDE_COLS);
            this->expect_same_bits(first, second);
        }

        TYPED_TEST(Unit_Level2SyclParityTest, SymvRepeatsToTheBit)
        {
            const auto first = this->run_symv(ExecutionMode::ForceSycl, K_LARGE_N, true);
            const auto second = this->run_symv(ExecutionMode::ForceSycl, K_LARGE_N, true);
            this->expect_same_bits(first, second);
        }
    } // namespace Test
} // namespace SushiBLAS
