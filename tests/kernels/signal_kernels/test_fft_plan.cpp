/**************************************************************************/
/* test_fft_plan.cpp                                                      */
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
 * @file test_fft_plan.cpp
 * @brief Tests that FftPlan cooks the radix-2 and Bluestein tables and sizes its scratch, and the butterfly and packed-bin steps of the transform.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_FftPlan, CooksTheBitReversalOfAPowerOfTwoLine)
        {
            const Kernels::FftPlan<double> plan(1, 8, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftLineView<double> row =
                plan.row_line(plan.values().data(), plan.indices().data());

            ASSERT_EQ(row.length, 8u);
            ASSERT_EQ(row.power_size, 8u);
            EXPECT_FALSE(row.is_bluestein());
            const std::vector<std::uint32_t> expected{0, 4, 2, 6, 1, 5, 3, 7};
            EXPECT_EQ(std::vector<std::uint32_t>(row.reversal, row.reversal + 8), expected);
        }

        /// @brief The absolute tolerance of a twiddle: one rounding of a value of order one.
        constexpr double K_TWIDDLE_TOLERANCE = absolute_tolerance<double>(1);

        /// @brief One twiddle of a line of eight and the unit root exp(-2 pi i k / 8) worked by hand.
        struct TwiddleRow
        {
                const char* name;
                std::size_t k;
                double real;
                double imaginary;
        };

        /// @brief Returns the four roots a line of eight keeps; sqrt(2) / 2 is 0.7071067811865476.
        static std::vector<TwiddleRow> twiddle_rows()
        {
            return {{"K0", 0, 1.0, 0.0},
                    {"K1", 1, 0.7071067811865476, -0.7071067811865476},
                    {"K2", 2, 0.0, -1.0},
                    {"K3", 3, -0.7071067811865476, -0.7071067811865476}};
        }

        /// @brief Names a row, which gtest takes as the test suffix.
        static std::string twiddle_row_name(const ::testing::TestParamInfo<TwiddleRow>& info)
        {
            return info.param.name;
        }

        /// @brief Reads one twiddle of a plan of eight.
        class Small_FftPlanTwiddles : public ::testing::TestWithParam<TwiddleRow>
        {
        };

        TEST_P(Small_FftPlanTwiddles, CooksTheTwiddlesAsUnitRoots)
        {
            const TwiddleRow& row = GetParam();
            const Kernels::FftPlan<double> plan(1, 8, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftLineView<double> line =
                plan.row_line(plan.values().data(), plan.indices().data());

            EXPECT_NEAR(line.twiddles[2 * row.k], row.real, K_TWIDDLE_TOLERANCE);
            EXPECT_NEAR(line.twiddles[2 * row.k + 1], row.imaginary, K_TWIDDLE_TOLERANCE);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftPlanTwiddles, ::testing::ValuesIn(twiddle_rows()),
                                 twiddle_row_name);

        TEST(Small_FftPlan, PadsANonPowerOfTwoLineForBluestein)
        {
            const Kernels::FftPlan<float> plan(1, 6, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftLineView<float> row =
                plan.row_line(plan.values().data(), plan.indices().data());

            EXPECT_EQ(row.length, 6u);
            // The padding is the least power of two at or above 2 * 6 - 1.
            EXPECT_EQ(row.power_size, 16u);
            EXPECT_TRUE(row.is_bluestein());
            EXPECT_EQ(plan.padded_scalars(3), 2u * 3u * 16u);
        }

        TEST(Small_FftPlan, PacksAnEvenRealRowIntoAHalfLine)
        {
            const Kernels::FftPlan<double> even(2, 8, Kernels::FftPairing::RealToComplex);
            const Kernels::FftPlan<double> odd(2, 7, Kernels::FftPairing::RealToComplex);

            EXPECT_TRUE(even.is_packed());
            EXPECT_EQ(even.row_line(even.values().data(), even.indices().data()).length, 4u);
            EXPECT_EQ(even.spectrum_columns(), 5u);
            EXPECT_NE(even.half_roots(even.values().data()), nullptr);
            EXPECT_FALSE(odd.is_packed());
            EXPECT_EQ(odd.row_line(odd.values().data(), odd.indices().data()).length, 7u);
            EXPECT_EQ(odd.spectrum_columns(), 4u);
            EXPECT_EQ(odd.half_roots(odd.values().data()), nullptr);
        }

        TEST(Small_FftPlan, CountsTheScalarsOfEachSide)
        {
            const Kernels::FftPlan<double> complex(3, 5, Kernels::FftPairing::ComplexToComplex);
            const Kernels::FftPlan<double> real(3, 6, Kernels::FftPairing::RealToComplex);

            EXPECT_EQ(complex.sample_scalars(), 30u);
            EXPECT_EQ(complex.spectrum_scalars(), 30u);
            EXPECT_EQ(complex.line_scalars(2), 0u);
            EXPECT_EQ(real.sample_scalars(), 18u);
            EXPECT_EQ(real.spectrum_scalars(), 24u);
            EXPECT_EQ(real.line_scalars(2), 2u * 2u * 3u * 3u);
        }

        TEST(Small_FftPlan, GivesOneRowNoColumnLine)
        {
            const Kernels::FftPlan<float> plan(1, 4, Kernels::FftPairing::ComplexToComplex);

            EXPECT_EQ(plan.column_line(plan.values().data(), plan.indices().data()).length, 0u);
        }

        TEST(Small_FftPlan, GivesAMultiRowPlanAColumnLineAsLongAsItsRows)
        {
            const Kernels::FftPlan<float> plan(3, 4, Kernels::FftPairing::ComplexToComplex);

            EXPECT_EQ(plan.column_line(plan.values().data(), plan.indices().data()).length, 3u);
        }

        /** @brief Holds a plan's shape, a batch, and the scratch it must report. */
        struct PlanScratchRow
        {
                const char* name;
                std::size_t rows;
                std::size_t columns;
                Kernels::FftPairing pairing;
                std::size_t batch;
                std::size_t copy_scalars;
                std::size_t bytes;
        };

        /** @brief Returns the rows with their scratch worked by hand for double. */
        static std::vector<PlanScratchRow> scratch_rows()
        {
            return {// 1 x 8 complex: 8 scalars and 8 indices, 8 * 8 + 8 * 4 = 96.
                    {"ComplexPowerOfTwo", 1, 8, Kernels::FftPairing::ComplexToComplex, 5, 0, 96},
                    // 1 x 3 complex: padded 8, (8 + 6 + 16 table scalars + 2 * 2 * 8 padded) * 8 + 8 * 4 = 528.
                    {"ComplexBluestein", 1, 3, Kernels::FftPairing::ComplexToComplex, 2, 0, 528},
                    // 1 x 8 real: packed lines of 4, (4 + 10 table scalars + 2 * 2 * 4 lines) * 8 + 4 * 4 = 256.
                    {"RealPackedOneRow", 1, 8, Kernels::FftPairing::RealToComplex, 2, 0, 256},
                    // 2 x 4 real: (2 + 2 + 6 table scalars + 24 lines + 3 * 12 copy) * 8 + 4 * 4 = 576.
                    {"RealPackedTwoRows", 2, 4, Kernels::FftPairing::RealToComplex, 3, 36, 576}};
        }

        /// @brief Names a row, which gtest takes as the test suffix.
        static std::string scratch_row_name(const ::testing::TestParamInfo<PlanScratchRow>& info)
        {
            return info.param.name;
        }

        class Small_FftPlanScratch : public ::testing::TestWithParam<PlanScratchRow>
        {
        };

        TEST_P(Small_FftPlanScratch, ReportsTheBytesOfTablesLinesCopyAndPaddedLines)
        {
            const PlanScratchRow row = GetParam();
            const Kernels::FftPlan<double> plan(row.rows, row.columns, row.pairing);

            EXPECT_EQ(plan.scratch_bytes(row.batch), row.bytes);
        }

        TEST_P(Small_FftPlanScratch, KeepsASpectrumCopyOnlyForARealInverseOverSeveralRows)
        {
            const PlanScratchRow row = GetParam();
            const Kernels::FftPlan<double> plan(row.rows, row.columns, row.pairing);

            EXPECT_EQ(plan.spectrum_copy_scalars(row.batch), row.copy_scalars);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftPlanScratch, ::testing::ValuesIn(scratch_rows()),
                                 scratch_row_name);

        /** @brief Holds a butterfly's inputs and the pair it must leave. */
        struct ButterflyStepRow
        {
                double u[2];
                double v[2];
                double wr;
                double wi;
                double new_u[2];
                double new_v[2];
        };

        class Small_FftButterfly : public ::testing::TestWithParam<ButterflyStepRow>
        {
        };

        TEST_P(Small_FftButterfly, LeavesThePairAsSumAndDifferenceOfUAndTheTwiddledV)
        {
            const ButterflyStepRow row = GetParam();
            double u[2] = {row.u[0], row.u[1]};
            double v[2] = {row.v[0], row.v[1]};

            Kernels::Detail::butterfly(u, v, row.wr, row.wi);

            EXPECT_EQ(u[0], row.new_u[0]);
            EXPECT_EQ(u[1], row.new_u[1]);
            EXPECT_EQ(v[0], row.new_v[0]);
            EXPECT_EQ(v[1], row.new_v[1]);
        }

        // v * w is (-4, 3), (3, 4) and (0.5, -1.5) for the three rows.
        INSTANTIATE_TEST_SUITE_P(, Small_FftButterfly,
                                 ::testing::Values(ButterflyStepRow{{1, 2}, {3, 4}, 0, 1, {-3, 5}, {5, -1}},
                                                   ButterflyStepRow{{1, 2}, {3, 4}, 1, 0, {4, 6}, {-2, -2}},
                                                   ButterflyStepRow{{0, 0}, {2, -1}, 0.5, -0.5,
                                                                {0.5, -1.5}, {-0.5, 1.5}}));

        /** @brief Holds the inputs of a packed-real bin step and the bin it must give. */
        struct PackedBinRow
        {
                double ar;
                double ai;
                double mr;
                double mi;
                double root_r;
                double root_i;
                double out_r;
                double out_i;
        };

        class Small_FftUnpackForwardBin : public ::testing::TestWithParam<PackedBinRow>
        {
        };

        TEST_P(Small_FftUnpackForwardBin, GivesTheEvenPartPlusTheTurnedOddPart)
        {
            const PackedBinRow row = GetParam();
            double out_r = 0;
            double out_i = 0;

            Kernels::Detail::unpack_forward_bin(row.ar, row.ai, row.mr, row.mi, row.root_r,
                                                row.root_i, out_r, out_i);

            EXPECT_EQ(out_r, row.out_r);
            EXPECT_EQ(out_i, row.out_i);
        }

        // z = (3, 1), m = (1, 1): even (2, 1), odd (2, 0) * (0, -0.5) = (0, -1); root (0, -1) turns it to (-1, 0).
        INSTANTIATE_TEST_SUITE_P(, Small_FftUnpackForwardBin,
                                 ::testing::Values(PackedBinRow{3, 1, 1, 1, 0, -1, 1, 1},
                                                   // Root (1, 0) keeps odd (0, -1): (2, 1) + (0, -1).
                                                   PackedBinRow{3, 1, 1, 1, 1, 0, 2, 0},
                                                   // z = (4, 0), m = 0: even (2, 0), odd (0, -2) turned to (-2, 0).
                                                   PackedBinRow{4, 0, 0, 0, 0, -1, 0, 0}));

        class Small_FftPackInverseBin : public ::testing::TestWithParam<PackedBinRow>
        {
        };

        TEST_P(Small_FftPackInverseBin, GivesTheEvenPartPlusITimesTheTwiddledOddPart)
        {
            const PackedBinRow row = GetParam();
            double out_r = 0;
            double out_i = 0;

            Kernels::Detail::pack_inverse_bin(row.ar, row.ai, row.mr, row.mi, row.root_r,
                                              row.root_i, out_r, out_i);

            EXPECT_EQ(out_r, row.out_r);
            EXPECT_EQ(out_i, row.out_i);
        }

        // a = (3, 1), m = (1, 1): even (4, 2); root (1, 0) leaves odd (2, 0), i * odd = (0, 2).
        INSTANTIATE_TEST_SUITE_P(, Small_FftPackInverseBin,
                                 ::testing::Values(PackedBinRow{3, 1, 1, 1, 1, 0, 4, 4},
                                                   // Root (0, -1) gives odd (0, 2), i * odd = (-2, 0).
                                                   PackedBinRow{3, 1, 1, 1, 0, -1, 2, 2}));
    } // namespace Test
} // namespace SushiBLAS
