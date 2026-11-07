/**************************************************************************/
/* test_segment_fold.cpp                                                  */
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
 * @file test_segment_fold.cpp
 * @brief Tests the fold of one segment of a ragged array, without a device.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/segment_fold.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Sum;
            using GraphPrimitives::Detail::fold_segment;

            /// @brief Holds an identity no sum of the values below equals.
            constexpr std::uint32_t K_IDENTITY = 7777u;

            /// @brief Names a segment of the ragged array and the fold it must give.
            struct SegmentRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the segment.
                std::size_t segment;
                /// @brief Holds the fold of the segment.
                std::uint32_t folded;
            };

            /** @brief Names a test case after its row. */
            std::string row_name(const ::testing::TestParamInfo<SegmentRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /// @brief Folds one segment of a ragged array of five values in four segments.
        class Small_GraphPrimitivesSegmentFold : public ::testing::TestWithParam<SegmentRow>
        {
        };

        /** @brief Tests that a segment folds its own values and an empty one gives the identity. */
        TEST_P(Small_GraphPrimitivesSegmentFold, SegmentFoldsItsOwnValuesOrGivesTheIdentity)
        {
            // Segments: [1, 2], an empty one, [3, 4, 5] and a single [6].
            const std::vector<std::uint32_t> values{1u, 2u, 3u, 4u, 5u, 6u};
            const std::vector<std::uint32_t> bounds{0u, 2u, 2u, 5u, 6u};

            EXPECT_EQ(fold_segment(Sum<std::uint32_t>{}, values.data(), bounds.data(),
                                   GetParam().segment, K_IDENTITY),
                      GetParam().folded);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesSegmentFold,
                                 ::testing::Values(SegmentRow{"TwoValues", 0, 3u},
                                                   SegmentRow{"EmptySegmentGivesTheIdentity", 1,
                                                              K_IDENTITY},
                                                   SegmentRow{"ThreeValues", 2, 12u},
                                                   SegmentRow{"SingleValue", 3, 6u}),
                                 row_name);

    } // namespace Test
} // namespace SushiBLAS
