/**************************************************************************/
/* test_blas_half_accumulation.cpp                                        */
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
 * @file test_blas_half_accumulation.cpp
 * @brief Tests that HALF GEMM, SYRK, GEMV, DOT, ASUM and NRM2 accumulate in FLOAT32 on both
 *        kernels, and that HALF TRSM and TRSV carry their solution in FLOAT32.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Reduction length of every case: 2^12, exact in HALF and twice where HALF sums of ones stall.
        constexpr int64_t K_REDUCTION_LENGTH = 4096;

        /// @brief Edge of the small output dimensions; the reduction length carries the test.
        constexpr int64_t K_OUTPUT_EDGE = 4;

        /// @brief The exact sum of K_REDUCTION_LENGTH products of one and one.
        constexpr double K_EXPECTED_SUM = 4096.0;

        /// @brief Half the HALF spacing at 4096 (spacing 4); a sum stalled at 2048 is far outside it.
        constexpr double K_HALF_ROUNDING_AT_SUM = 2.0;

        /// @brief The exact 2-norm of K_REDUCTION_LENGTH ones: sqrt(4096).
        constexpr double K_EXPECTED_NORM = 64.0;

        /// @brief Half the HALF spacing at 64 (spacing 1/16); a norm stalled at sqrt(2048) is not.
        constexpr double K_HALF_ROUNDING_AT_NORM = 0.03125;

        /// @brief Order of the unit-lower system the substitution cases solve.
        constexpr int64_t K_SYSTEM_ORDER = 64;

        /// @brief Right-hand sides the TRSM cases solve in one call.
        constexpr int64_t K_RIGHT_HAND_SIDES = 3;

        /** @brief Selects the order a backward step subtracts its solved elements in. */
        enum class BackwardOrder : std::uint8_t
        {
            Ascending, /**< From the element after the step up to the last one. */
            Descending /**< From the last element down to the one after the step. */
        };

        /**
         * @brief Runs HALF BLAS calls in a chosen execution mode, on all-ones operands and on a
         *        unit-lower system.
         */
        class Unit_BlasHalfAccumulationTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a HALF tensor of dims with every element one. */
                [[nodiscard]] Tensor ones(std::initializer_list<int64_t> dims)
                {
                    Tensor t = engine->create_tensor(dims, Core::DataType::Half);
                    const std::vector<sycl::half> values(static_cast<std::size_t>(t.num_elements),
                                                         sycl::half(1.0f));
                    SushiBLAS::IO(*engine).from_vector(t, values);
                    return t;
                }

                /** @brief Returns a HALF tensor of dims with every element zero. */
                [[nodiscard]] Tensor zeros(std::initializer_list<int64_t> dims)
                {
                    Tensor t = engine->create_tensor(dims, Core::DataType::Half);
                    const std::vector<sycl::half> values(static_cast<std::size_t>(t.num_elements),
                                                         sycl::half(0.0f));
                    SushiBLAS::IO(*engine).from_vector(t, values);
                    return t;
                }

                /** @brief Returns a HALF tensor of dims holding values. */
                [[nodiscard]] Tensor half_tensor(std::initializer_list<int64_t> dims,
                                                 const std::vector<sycl::half>& values)
                {
                    Tensor t = engine->create_tensor(dims, Core::DataType::Half);
                    SushiBLAS::IO(*engine).from_vector(t, values);
                    return t;
                }

                /** @brief Returns a row-major unit-lower A of order K_SYSTEM_ORDER, zero above. */
                [[nodiscard]] static std::vector<sycl::half> unit_lower_values()
                {
                    const std::size_t order = static_cast<std::size_t>(K_SYSTEM_ORDER);
                    std::vector<sycl::half> values(order * order, sycl::half(0.0f));
                    for (std::size_t i = 0; i < order; ++i)
                    {
                        for (std::size_t k = 0; k < i; ++k)
                        {
                            const int64_t step = static_cast<int64_t>((i * 7u + k * 13u) % 17u) - 8;
                            values[i * order + k] = sycl::half(static_cast<float>(step) / 64.0f);
                        }
                        values[i * order + i] = sycl::half(1.0f);
                    }
                    return values;
                }

                /** @brief Returns count right-hand-side elements, each a multiple of 1/8. */
                [[nodiscard]] static std::vector<sycl::half> right_hand_side_values(int64_t count)
                {
                    std::vector<sycl::half> values(static_cast<std::size_t>(count));
                    for (std::size_t i = 0; i < values.size(); ++i)
                    {
                        const int64_t step = static_cast<int64_t>((i * 5u) % 23u) - 11;
                        values[i] = sycl::half(static_cast<float>(step) / 8.0f);
                    }
                    return values;
                }

                /**
                 * @brief Returns the host solution of the unit-lower system for a rows x cols
                 *        right-hand side, each element rounded once to HALF.
                 * @tparam Sum The type the sum of a step is formed in.
                 * @tparam Carried The type a solved element is kept in between steps.
                 * @param backward The order of the right side's steps; the left side ignores it.
                 */
                template<typename Sum, typename Carried>
                [[nodiscard]] static std::vector<sycl::half> substitute(Core::Side side,
                                                                        int64_t rows, int64_t cols,
                                                                        BackwardOrder backward)
                {
                    const std::vector<sycl::half> a = unit_lower_values();
                    const std::vector<sycl::half> b = right_hand_side_values(rows * cols);
                    std::vector<Carried> carried(b.size());
                    std::vector<sycl::half> solution(b.size());

                    const auto at_a = [](int64_t r, int64_t c)
                    {
                        return static_cast<std::size_t>(r * K_SYSTEM_ORDER + c);
                    };
                    const auto at_b = [cols](int64_t r, int64_t c)
                    {
                        return static_cast<std::size_t>(r * cols + c);
                    };
                    const auto term = [&](std::size_t a_index, std::size_t x_index)
                    {
                        return static_cast<Sum>(static_cast<float>(a[a_index])) *
                               static_cast<Sum>(static_cast<float>(carried[x_index]));
                    };
                    const auto store = [&](std::size_t index, Sum sum)
                    {
                        const float wide = static_cast<float>(sum);
                        carried[index] = static_cast<Carried>(wide);
                        solution[index] = sycl::half(wide);
                    };

                    if (side == Core::Side::Left)
                    {
                        for (int64_t j = 0; j < cols; ++j)
                        {
                            for (int64_t i = 0; i < rows; ++i)
                            {
                                Sum sum = static_cast<Sum>(static_cast<float>(b[at_b(i, j)]));
                                for (int64_t k = 0; k < i; ++k)
                                    sum -= term(at_a(i, k), at_b(k, j));
                                store(at_b(i, j), sum);
                            }
                        }
                    }
                    else
                    {
                        for (int64_t i = 0; i < rows; ++i)
                        {
                            for (int64_t p = cols - 1; p >= 0; --p)
                            {
                                Sum sum = static_cast<Sum>(static_cast<float>(b[at_b(i, p)]));
                                if (backward == BackwardOrder::Descending)
                                {
                                    for (int64_t q = cols - 1; q > p; --q)
                                        sum -= term(at_a(q, p), at_b(i, q));
                                }
                                else
                                {
                                    for (int64_t q = p + 1; q < cols; ++q)
                                        sum -= term(at_a(q, p), at_b(i, q));
                                }
                                store(at_b(i, p), sum);
                            }
                        }
                    }
                    return solution;
                }

                /** @brief Returns the X of the unit-lower TRSM on side, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_trsm(ExecutionMode mode, Core::Side side,
                                                               int64_t rows, int64_t cols)
                {
                    engine->set_mode(mode);
                    Tensor A = half_tensor({K_SYSTEM_ORDER, K_SYSTEM_ORDER}, unit_lower_values());
                    Tensor B = half_tensor({rows, cols}, right_hand_side_values(rows * cols));
                    SushiBLAS::BLASOps(*engine).trsm(A, B, side, Core::Uplo::Lower,
                                                     Core::Transpose::NoTrans, Core::Diag::Unit);
                    engine->execute();
                    return SushiBLAS::IO(*engine).to_vector<sycl::half>(B);
                }

                /** @brief Returns the x of the unit-lower TRSV, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_trsv(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    Tensor A = half_tensor({K_SYSTEM_ORDER, K_SYSTEM_ORDER}, unit_lower_values());
                    Tensor b = half_tensor({K_SYSTEM_ORDER},
                                           right_hand_side_values(K_SYSTEM_ORDER));
                    SushiBLAS::BLASOps(*engine).trsv(A, b, Core::Uplo::Lower,
                                                     Core::Transpose::NoTrans, Core::Diag::Unit);
                    engine->execute();
                    return SushiBLAS::IO(*engine).to_vector<sycl::half>(b);
                }

                /** @brief Returns the bit pattern of a HALF value. */
                [[nodiscard]] static std::uint16_t bits_of(sycl::half value) noexcept
                {
                    static_assert(sizeof(sycl::half) == sizeof(std::uint16_t),
                                  "A HALF element is 16 bits.");
                    std::uint16_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    return bits;
                }

                /** @brief Returns whether two solutions differ in the bits of any element. */
                [[nodiscard]] static bool differs(const std::vector<sycl::half>& first,
                                                  const std::vector<sycl::half>& second)
                {
                    return first.size() != second.size() ||
                           std::memcmp(first.data(), second.data(),
                                       first.size() * sizeof(sycl::half)) != 0;
                }

                /** @brief Expects two solutions to hold the same bits, element by element. */
                static void expect_same_bits(const std::vector<sycl::half>& actual,
                                             const std::vector<sycl::half>& expected)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        EXPECT_EQ(bits_of(actual[i]), bits_of(expected[i])) << "index " << i;
                }

                /**
                 * @brief Expects each mode to equal its host substitution bit for bit, and neither
                 *        substitution to equal the one that rounds the solution between steps.
                 * @param run Callable `(ExecutionMode)` that returns the solution of that mode.
                 */
                template<typename Run>
                void expect_carried_in_float(Core::Side side, int64_t rows, int64_t cols, Run run)
                {
                    const auto summed_in_float =
                        substitute<float, float>(side, rows, cols, BackwardOrder::Descending);
                    const auto summed_in_double =
                        substitute<double, float>(side, rows, cols, BackwardOrder::Ascending);
                    const auto carried_in_half =
                        substitute<float, sycl::half>(side, rows, cols, BackwardOrder::Descending);

                    EXPECT_TRUE(differs(summed_in_float, carried_in_half));
                    EXPECT_TRUE(differs(summed_in_double, carried_in_half));
                    expect_same_bits(run(ExecutionMode::ForceSycl), summed_in_float);
                    expect_same_bits(run(ExecutionMode::ForceReference), summed_in_double);
                }

                /** @brief Returns C = A * B for all-ones A and B, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_gemm(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    Tensor A = ones({K_OUTPUT_EDGE, K_REDUCTION_LENGTH});
                    Tensor B = ones({K_REDUCTION_LENGTH, K_OUTPUT_EDGE});
                    Tensor C = zeros({K_OUTPUT_EDGE, K_OUTPUT_EDGE});
                    SushiBLAS::BLASOps(*engine).gemm(A, B, C, Core::Transpose::NoTrans,
                                                     Core::Transpose::NoTrans, 1.0, 0.0);
                    engine->execute();
                    return SushiBLAS::IO(*engine).to_vector<sycl::half>(C);
                }

                /** @brief Returns the lower triangle of C = A * A^T for an all-ones A, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_syrk(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    Tensor A = ones({K_OUTPUT_EDGE, K_REDUCTION_LENGTH});
                    Tensor C = zeros({K_OUTPUT_EDGE, K_OUTPUT_EDGE});
                    SushiBLAS::BLASOps(*engine).syrk(A, C, Core::Uplo::Lower,
                                                     Core::Transpose::NoTrans, 1.0, 0.0);
                    engine->execute();

                    const std::vector<sycl::half> full =
                        SushiBLAS::IO(*engine).to_vector<sycl::half>(C);
                    std::vector<sycl::half> lower;
                    for (int64_t i = 0; i < K_OUTPUT_EDGE; ++i)
                        for (int64_t j = 0; j <= i; ++j)
                            lower.push_back(full[static_cast<std::size_t>(i * K_OUTPUT_EDGE + j)]);
                    return lower;
                }

                /** @brief Returns y = A * x for all-ones A and x, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_gemv(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    Tensor A = ones({K_OUTPUT_EDGE, K_REDUCTION_LENGTH});
                    Tensor x = ones({K_REDUCTION_LENGTH});
                    Tensor y = zeros({K_OUTPUT_EDGE});
                    SushiBLAS::BLASOps(*engine).gemv(A, x, y, Core::Transpose::NoTrans, 1.0, 0.0);
                    engine->execute();
                    return SushiBLAS::IO(*engine).to_vector<sycl::half>(y);
                }

                /**
                 * @brief Returns the HALF scalar a Level 1 reduction writes for an all-ones
                 *        vector, computed in mode.
                 * @param call Callable `(Tensor& x, Tensor& result)` that records the reduction.
                 */
                template<typename Call>
                [[nodiscard]] double run_level1(ExecutionMode mode, Call call)
                {
                    engine->set_mode(mode);
                    Tensor x = ones({K_REDUCTION_LENGTH});
                    Tensor result = zeros({1});
                    call(x, result);
                    engine->execute();
                    const sycl::half value =
                        SushiBLAS::IO(*engine).to_vector<sycl::half>(result)[0];
                    return static_cast<double>(static_cast<float>(value));
                }

                /** @brief Expects every element to equal K_EXPECTED_SUM within K_HALF_ROUNDING_AT_SUM. */
                static void expect_full_sum(const std::vector<sycl::half>& actual)
                {
                    ASSERT_FALSE(actual.empty());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        EXPECT_NEAR(static_cast<double>(static_cast<float>(actual[i])),
                                    K_EXPECTED_SUM, K_HALF_ROUNDING_AT_SUM) << "index " << i;
                }

                /** @brief Expects the SYCL result to match the reference result element by element. */
                static void expect_match(const std::vector<sycl::half>& actual,
                                         const std::vector<sycl::half>& expected)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        EXPECT_NEAR(static_cast<double>(static_cast<float>(actual[i])),
                                    static_cast<double>(static_cast<float>(expected[i])),
                                    K_HALF_ROUNDING_AT_SUM) << "index " << i;
                }
        };

        TEST_F(Unit_BlasHalfAccumulationTest, GemmSumsEveryTermOfALongReduction)
        {
            const auto expected = run_gemm(ExecutionMode::ForceReference);
            const auto actual = run_gemm(ExecutionMode::ForceSycl);
            expect_full_sum(expected);
            expect_full_sum(actual);
            expect_match(actual, expected);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, SyrkSumsEveryTermOfALongReduction)
        {
            const auto expected = run_syrk(ExecutionMode::ForceReference);
            const auto actual = run_syrk(ExecutionMode::ForceSycl);
            expect_full_sum(expected);
            expect_full_sum(actual);
            expect_match(actual, expected);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, GemvSumsEveryTermOfALongReduction)
        {
            const auto expected = run_gemv(ExecutionMode::ForceReference);
            const auto actual = run_gemv(ExecutionMode::ForceSycl);
            expect_full_sum(expected);
            expect_full_sum(actual);
            expect_match(actual, expected);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, DotSumsEveryTermOfALongReduction)
        {
            auto dot = [this](Tensor& x, Tensor& result)
            {
                SushiBLAS::BLASOps(*engine).dot(x, x, result);
            };
            EXPECT_NEAR(run_level1(ExecutionMode::ForceReference, dot), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
            EXPECT_NEAR(run_level1(ExecutionMode::ForceSycl, dot), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, AsumSumsEveryTermOfALongReduction)
        {
            auto asum = [this](Tensor& x, Tensor& result)
            {
                SushiBLAS::BLASOps(*engine).asum(x, result);
            };
            EXPECT_NEAR(run_level1(ExecutionMode::ForceReference, asum), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
            EXPECT_NEAR(run_level1(ExecutionMode::ForceSycl, asum), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, Nrm2SumsEverySquareOfALongReduction)
        {
            auto nrm2 = [this](Tensor& x, Tensor& result)
            {
                SushiBLAS::BLASOps(*engine).nrm2(x, result);
            };
            EXPECT_NEAR(run_level1(ExecutionMode::ForceReference, nrm2), K_EXPECTED_NORM,
                        K_HALF_ROUNDING_AT_NORM);
            EXPECT_NEAR(run_level1(ExecutionMode::ForceSycl, nrm2), K_EXPECTED_NORM,
                        K_HALF_ROUNDING_AT_NORM);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, HalfTrsmCarriesTheSolutionInFloat)
        {
            for (const Core::Side side : {Core::Side::Left, Core::Side::Right})
            {
                const bool left = side == Core::Side::Left;
                const int64_t rows = left ? K_SYSTEM_ORDER : K_RIGHT_HAND_SIDES;
                const int64_t cols = left ? K_RIGHT_HAND_SIDES : K_SYSTEM_ORDER;
                expect_carried_in_float(side, rows, cols, [&](ExecutionMode mode)
                {
                    return run_trsm(mode, side, rows, cols);
                });
            }
        }

        TEST_F(Unit_BlasHalfAccumulationTest, HalfTrsvCarriesTheSolutionInFloat)
        {
            expect_carried_in_float(Core::Side::Left, K_SYSTEM_ORDER, 1, [this](ExecutionMode mode)
            {
                return run_trsv(mode);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
