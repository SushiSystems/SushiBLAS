/**************************************************************************/
/* test_fft_line_index.cpp                                                */
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
 * @file test_fft_line_index.cpp
 * @brief Tests the FFT line layouts, the butterfly positions and the spectrum mirroring against nested loops and hand values.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/signal_kernels/fft_line_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds @p lines contiguous lines of @p length complex elements. */
        struct ContiguousRow
        {
            std::size_t lines;
            std::size_t length;
        };

        /** @brief Checks that line l, element k of a contiguous layout sits at l * length + k. */
        static void expect_contiguous_offsets(const ContiguousRow& row)
        {
            const Indexing::FftLineLayout layout = Indexing::contiguous_lines(row.lines, row.length);
            EXPECT_EQ(layout.lines, row.lines);
            for (std::size_t l = 0; l < row.lines; ++l)
            {
                for (std::size_t k = 0; k < row.length; ++k)
                    EXPECT_EQ(layout.at(l, k), l * row.length + k) << "line " << l << " element " << k;
            }
        }

        class Small_FftLineIndexContiguous : public ::testing::TestWithParam<ContiguousRow>
        {
        };

        TEST_P(Small_FftLineIndexContiguous, ContiguousLinesSitBackToBack)
        {
            expect_contiguous_offsets(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexContiguous,
                                 ::testing::Values(ContiguousRow{0, 4}, ContiguousRow{1, 1},
                                                   ContiguousRow{2, 3}, ContiguousRow{3, 2},
                                                   ContiguousRow{5, 5}, ContiguousRow{9, 8}));

        /** @brief Holds the spectra of a batch: how many, their rows and their complex columns. */
        struct ColumnLayoutRow
        {
            std::size_t batch;
            std::size_t rows;
            std::size_t width;
        };

        /** @brief Checks that column c of spectrum b, row r sits at (b * rows + r) * width + c. */
        static void expect_column_offsets(const ColumnLayoutRow& row)
        {
            const Indexing::FftLineLayout layout =
                Indexing::column_layout(row.batch, row.rows, row.width);
            EXPECT_EQ(layout.lines, row.batch * row.width);
            for (std::size_t b = 0; b < row.batch; ++b)
            {
                for (std::size_t c = 0; c < row.width; ++c)
                {
                    for (std::size_t r = 0; r < row.rows; ++r)
                        EXPECT_EQ(layout.at(b * row.width + c, r), (b * row.rows + r) * row.width + c)
                            << "spectrum " << b << " column " << c << " row " << r;
                }
            }
        }

        class Small_FftLineIndexColumns : public ::testing::TestWithParam<ColumnLayoutRow>
        {
        };

        TEST_P(Small_FftLineIndexColumns, ColumnLinesWalkDownTheRowsOfEachSpectrum)
        {
            expect_column_offsets(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexColumns,
                                 ::testing::Values(ColumnLayoutRow{0, 3, 4}, ColumnLayoutRow{1, 1, 1},
                                                   ColumnLayoutRow{1, 2, 3}, ColumnLayoutRow{2, 3, 4},
                                                   ColumnLayoutRow{3, 5, 2}, ColumnLayoutRow{2, 9, 5}));

        /** @brief Holds the lines and the width of a flat work grid. */
        struct FlatGrid
        {
            std::size_t lines;
            std::size_t width;
        };

        /** @brief Checks fft_flat_coord against nested loops at every flat index. */
        static void expect_flat_split_follows_the_loops(const FlatGrid& grid)
        {
            std::size_t flat = 0;
            for (std::size_t line = 0; line < grid.lines; ++line)
            {
                for (std::size_t element = 0; element < grid.width; ++element, ++flat)
                {
                    const Indexing::FftFlatCoord at = Indexing::fft_flat_coord(flat, grid.width);
                    EXPECT_EQ(at.line, line) << "flat " << flat;
                    EXPECT_EQ(at.element, element) << "flat " << flat;
                }
            }
        }

        class Small_FftLineIndexFlat : public ::testing::TestWithParam<FlatGrid>
        {
        };

        TEST_P(Small_FftLineIndexFlat, FlatCoordFollowsTheNestedLoops)
        {
            expect_flat_split_follows_the_loops(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexFlat,
                                 ::testing::Values(FlatGrid{0, 3}, FlatGrid{1, 1}, FlatGrid{2, 3},
                                                   FlatGrid{3, 2}, FlatGrid{5, 5}, FlatGrid{9, 1},
                                                   FlatGrid{1, 9}));

        /** @brief Holds one butterfly of a pass and the positions it must give. */
        struct ButterflyPairRow
        {
            std::size_t size;
            std::size_t span;
            std::size_t b;
            std::size_t upper;
            std::size_t lower;
            std::size_t twiddle;
        };

        class Small_FftLineIndexButterfly : public ::testing::TestWithParam<ButterflyPairRow>
        {
        };

        TEST_P(Small_FftLineIndexButterfly, ButterflyNumberPairsAnElementWithItsPartnerAHalfSpanAway)
        {
            const ButterflyPairRow row = GetParam();

            const Indexing::ButterflyPositions at = Indexing::butterfly_number(row.b, row.span, row.size);

            EXPECT_EQ(at.upper, row.upper);
            EXPECT_EQ(at.lower, row.lower);
            EXPECT_EQ(at.twiddle, row.twiddle);
        }

        // Size 8, span 4: half 2, stride 2, blocks at 0 and 4; size 4, span 2: half 1, root 0 each.
        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexButterfly,
                                 ::testing::Values(ButterflyPairRow{2, 2, 0, 0, 1, 0},
                                                   ButterflyPairRow{4, 2, 0, 0, 1, 0},
                                                   ButterflyPairRow{4, 2, 1, 2, 3, 0},
                                                   ButterflyPairRow{8, 4, 0, 0, 2, 0},
                                                   ButterflyPairRow{8, 4, 1, 1, 3, 2},
                                                   ButterflyPairRow{8, 4, 2, 4, 6, 0},
                                                   ButterflyPairRow{8, 4, 3, 5, 7, 2},
                                                   ButterflyPairRow{8, 8, 3, 3, 7, 3}));

        /** @brief Holds the radix-2 size of a line. */
        struct PassRow
        {
            std::size_t size;
        };

        /** @brief Checks butterfly_number against the nested loops of a pass, for every pass of a line. */
        static void expect_butterflies_follow_the_loops(const PassRow& row)
        {
            for (std::size_t span = 2; span <= row.size; span <<= 1)
            {
                const std::size_t half = span / 2;
                std::size_t b = 0;
                for (std::size_t start = 0; start < row.size; start += span)
                {
                    for (std::size_t k = 0; k < half; ++k, ++b)
                    {
                        const Indexing::ButterflyPositions at =
                            Indexing::butterfly_number(b, span, row.size);
                        EXPECT_EQ(at.upper, start + k) << "span " << span << " b " << b;
                        EXPECT_EQ(at.lower, start + k + half) << "span " << span << " b " << b;
                        EXPECT_EQ(at.twiddle, k * (row.size / span)) << "span " << span << " b " << b;
                    }
                }
            }
        }

        class Small_FftLineIndexPasses : public ::testing::TestWithParam<PassRow>
        {
        };

        TEST_P(Small_FftLineIndexPasses, ButterflyNumberFollowsTheNestedLoopsOfEveryPass)
        {
            expect_butterflies_follow_the_loops(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexPasses,
                                 ::testing::Values(PassRow{2}, PassRow{4}, PassRow{8},
                                                   PassRow{16}));

        /** @brief Holds one is_padding_lane call. */
        struct PaddingRow
        {
            std::size_t k;
            std::size_t length;
            bool expected;
        };

        class Small_FftLineIndexPadding : public ::testing::TestWithParam<PaddingRow>
        {
        };

        TEST_P(Small_FftLineIndexPadding, ALaneIsPaddingFromTheLengthOn)
        {
            const PaddingRow row = GetParam();

            EXPECT_EQ(Indexing::is_padding_lane(row.k, row.length), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexPadding,
                                 ::testing::Values(PaddingRow{0, 0, true}, PaddingRow{0, 1, false},
                                                   PaddingRow{1, 1, true}, PaddingRow{2, 3, false},
                                                   PaddingRow{3, 3, true}, PaddingRow{7, 3, true}));

        /** @brief Holds one mirror_bin call. */
        struct MirrorRow
        {
            std::size_t k;
            std::size_t half;
            std::size_t expected;
        };

        class Small_FftLineIndexMirror : public ::testing::TestWithParam<MirrorRow>
        {
        };

        TEST_P(Small_FftLineIndexMirror, TheMirrorOfABinIsHalfMinusItWrappedAtHalf)
        {
            const MirrorRow row = GetParam();

            EXPECT_EQ(Indexing::mirror_bin(row.k, row.half), row.expected);
        }

        // Half 4: 0 pairs with itself, 1 with 3, 2 with itself, 3 with 1, and bin 4 wraps to 0.
        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexMirror,
                                 ::testing::Values(MirrorRow{0, 1, 0}, MirrorRow{1, 1, 0},
                                                   MirrorRow{0, 2, 0}, MirrorRow{1, 2, 1},
                                                   MirrorRow{2, 2, 0}, MirrorRow{0, 4, 0},
                                                   MirrorRow{1, 4, 3}, MirrorRow{2, 4, 2},
                                                   MirrorRow{3, 4, 1}, MirrorRow{4, 4, 0}));

        /** @brief Holds one hermitian_source call. */
        struct HermitianRow
        {
            std::size_t k;
            std::size_t width;
            std::size_t columns;
            std::size_t bin;
            bool conjugated;
        };

        class Small_FftLineIndexHermitian : public ::testing::TestWithParam<HermitianRow>
        {
        };

        TEST_P(Small_FftLineIndexHermitian, TheMissingBinsAreTheConjugatesOfTheStoredOnes)
        {
            const HermitianRow row = GetParam();

            const Indexing::HermitianSource from = Indexing::hermitian_source(row.k, row.width, row.columns);

            EXPECT_EQ(from.bin, row.bin);
            EXPECT_EQ(from.conjugated, row.conjugated);
        }

        // Eight columns store bins 0 .. 4 and mirror 3, 2, 1; seven store 0 .. 3 and mirror 3, 2, 1.
        INSTANTIATE_TEST_SUITE_P(, Small_FftLineIndexHermitian,
                                 ::testing::Values(HermitianRow{0, 1, 1, 0, false},
                                                   HermitianRow{0, 2, 2, 0, false},
                                                   HermitianRow{1, 2, 2, 1, false},
                                                   HermitianRow{4, 5, 8, 4, false},
                                                   HermitianRow{5, 5, 8, 3, true},
                                                   HermitianRow{6, 5, 8, 2, true},
                                                   HermitianRow{7, 5, 8, 1, true},
                                                   HermitianRow{3, 4, 7, 3, false},
                                                   HermitianRow{4, 4, 7, 3, true},
                                                   HermitianRow{6, 4, 7, 1, true}));
    } // namespace Test
} // namespace SushiBLAS
