/**************************************************************************/
/* test_convolve_index.cpp                                                */
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
 * @file test_convolve_index.cpp
 * @brief Tests the output length, the sample split and the tap sources of the valid-mode convolution.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/signal_kernels/convolve_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds one convolve_out_length call. */
        struct OutLengthRow
        {
            int64_t signal_length;
            int64_t tap_count;
            int64_t expected;
        };

        class Small_ConvolveIndexLength : public ::testing::TestWithParam<OutLengthRow>
        {
        };

        TEST_P(Small_ConvolveIndexLength, AValidConvolutionWritesLengthMinusTapsPlusOneSamples)
        {
            const OutLengthRow row = GetParam();

            EXPECT_EQ(Indexing::convolve_out_length(row.signal_length, row.tap_count), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ConvolveIndexLength,
                                 ::testing::Values(OutLengthRow{1, 1, 1}, OutLengthRow{2, 1, 2},
                                                   OutLengthRow{3, 3, 1}, OutLengthRow{5, 3, 3},
                                                   OutLengthRow{9, 2, 8}, OutLengthRow{9, 9, 1},
                                                   OutLengthRow{5, 1, 5}));

        /** @brief Holds the rows of a batch and the samples each writes. */
        struct SampleGrid
        {
            int64_t rows;
            int64_t out_length;
        };

        /** @brief Returns the output samples in storage order, sample fastest, listed by nested loops. */
        static std::vector<Indexing::ConvolveCoord> listed_samples(const SampleGrid& grid)
        {
            std::vector<Indexing::ConvolveCoord> samples;
            for (int64_t row = 0; row < grid.rows; ++row)
            {
                for (int64_t n = 0; n < grid.out_length; ++n)
                    samples.push_back({row, n});
            }
            return samples;
        }

        /** @brief Checks convolve_coord against the listing at every flat index. */
        static void expect_split_follows_the_listing(const SampleGrid& grid)
        {
            const std::vector<Indexing::ConvolveCoord> listed = listed_samples(grid);
            for (std::size_t i = 0; i < listed.size(); ++i)
            {
                const Indexing::ConvolveCoord at =
                    Indexing::convolve_coord(static_cast<int64_t>(i), grid.out_length);
                EXPECT_EQ(at.row, listed[i].row) << "flat " << i;
                EXPECT_EQ(at.n, listed[i].n) << "flat " << i;
            }
        }

        class Small_ConvolveIndexGrids : public ::testing::TestWithParam<SampleGrid>
        {
        };

        TEST_P(Small_ConvolveIndexGrids, ConvolveCoordFollowsTheNestedLoops)
        {
            expect_split_follows_the_listing(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ConvolveIndexGrids,
                                 ::testing::Values(SampleGrid{0, 3}, SampleGrid{1, 1},
                                                   SampleGrid{2, 3}, SampleGrid{3, 2},
                                                   SampleGrid{5, 5}, SampleGrid{9, 1},
                                                   SampleGrid{1, 9}));

        /** @brief Holds one output sample and the taps that touch it. */
        struct ConvolveSourceRow
        {
            int64_t n;
            int64_t tap_count;
        };

        /** @brief Returns the samples taps 0 .. K - 1 read for sample n: n + K - 1 down to n. */
        static std::vector<int64_t> descending_run(const ConvolveSourceRow& row)
        {
            std::vector<int64_t> run;
            for (int64_t sample = row.n + row.tap_count - 1; sample >= row.n; --sample)
                run.push_back(sample);
            return run;
        }

        /** @brief Returns the samples convolve_source gives for taps 0 .. K - 1. */
        static std::vector<int64_t> sources_of_taps(const ConvolveSourceRow& row)
        {
            std::vector<int64_t> run;
            for (int64_t k = 0; k < row.tap_count; ++k)
                run.push_back(Indexing::convolve_source(row.n, row.tap_count, k));
            return run;
        }

        class Small_ConvolveIndexSources : public ::testing::TestWithParam<ConvolveSourceRow>
        {
        };

        TEST_P(Small_ConvolveIndexSources, TapsReadTheWindowOfTheSampleBackwards)
        {
            EXPECT_EQ(sources_of_taps(GetParam()), descending_run(GetParam()));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ConvolveIndexSources,
                                 ::testing::Values(ConvolveSourceRow{0, 1}, ConvolveSourceRow{4, 1},
                                                   ConvolveSourceRow{0, 2}, ConvolveSourceRow{3, 3},
                                                   ConvolveSourceRow{0, 5}, ConvolveSourceRow{2, 9},
                                                   ConvolveSourceRow{7, 2}));
    } // namespace Test
} // namespace SushiBLAS
