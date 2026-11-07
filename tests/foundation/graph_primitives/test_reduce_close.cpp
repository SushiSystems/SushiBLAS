/**************************************************************************/
/* test_reduce_close.cpp                                                  */
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
 * @file test_reduce_close.cpp
 * @brief Tests the identity an empty reduction writes, without a device.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/reduce_close.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            namespace Close = GraphPrimitives::Detail;

            /// @brief Holds the identity of a minimum over uint32.
            constexpr std::uint32_t K_IDENTITY = 0xFFFFFFFFu;

            /// @brief Holds the result the folded levels left in the output.
            constexpr std::uint32_t K_FOLDED = 17u;

            /// @brief Names a live count and the result the closing node leaves.
            struct CloseRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the live element count of the run.
                std::size_t live_n;
                /// @brief Holds the result the output must hold afterwards.
                std::uint32_t result;
            };

            /** @brief Names a test case after its row. */
            std::string row_name(const ::testing::TestParamInfo<CloseRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Tests that the identity replaces whatever the output held. */
        TEST(Small_GraphPrimitivesReduceClose, IdentityReplacesTheOutput)
        {
            std::uint32_t out = K_FOLDED;

            Close::store_identity(&out, K_IDENTITY);

            EXPECT_EQ(out, K_IDENTITY);
        }

        /// @brief Closes a reduction at one live count.
        class Small_GraphPrimitivesReduceCloseLive : public ::testing::TestWithParam<CloseRow>
        {
        };

        /** @brief Tests that only a live count of zero writes the identity. */
        TEST_P(Small_GraphPrimitivesReduceCloseLive, OnlyAnEmptyRunWritesTheIdentity)
        {
            std::uint32_t out = K_FOLDED;

            Close::close_empty_reduction(&out, GetParam().live_n, K_IDENTITY);

            EXPECT_EQ(out, GetParam().result);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesReduceCloseLive,
                                 ::testing::Values(CloseRow{"NoLiveValue", 0, K_IDENTITY},
                                                   CloseRow{"OneLiveValue", 1, K_FOLDED},
                                                   CloseRow{"ManyLiveValues", 1000000, K_FOLDED}),
                                 row_name);

    } // namespace Test
} // namespace SushiBLAS
