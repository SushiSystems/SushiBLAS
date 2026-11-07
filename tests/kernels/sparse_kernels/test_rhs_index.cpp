/**************************************************************************/
/* test_rhs_index.cpp                                                     */
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
 * @file test_rhs_index.cpp
 * @brief Tests the split of a flat product index into a matrix row and a right-hand side against nested loops.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_kernels/rhs_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds the extents of a product: matrix rows and right-hand sides. */
        struct ProductGrid
        {
            int64_t rows;
            int64_t nrhs;
        };

        /** @brief Returns the elements in storage order, right-hand side fastest, listed by nested loops. */
        static std::vector<Indexing::RhsCoord> listed_elements(const ProductGrid& grid)
        {
            std::vector<Indexing::RhsCoord> elements;
            for (int64_t row = 0; row < grid.rows; ++row)
            {
                for (int64_t rhs = 0; rhs < grid.nrhs; ++rhs)
                    elements.push_back({row, rhs});
            }
            return elements;
        }

        /** @brief Checks rhs_coord against the listing at every flat index. */
        static void expect_split_follows_the_listing(const ProductGrid& grid)
        {
            const std::vector<Indexing::RhsCoord> listed = listed_elements(grid);
            for (std::size_t i = 0; i < listed.size(); ++i)
            {
                const Indexing::RhsCoord at = Indexing::rhs_coord(static_cast<int64_t>(i), grid.nrhs);
                EXPECT_EQ(at.row, listed[i].row) << "flat " << i;
                EXPECT_EQ(at.rhs, listed[i].rhs) << "flat " << i;
            }
        }

        class Small_RhsIndexGrids : public ::testing::TestWithParam<ProductGrid>
        {
        };

        TEST_P(Small_RhsIndexGrids, RhsCoordFollowsTheNestedLoops)
        {
            expect_split_follows_the_listing(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RhsIndexGrids,
                                 ::testing::Values(ProductGrid{0, 3}, ProductGrid{1, 1},
                                                   ProductGrid{2, 3}, ProductGrid{3, 2},
                                                   ProductGrid{5, 5}, ProductGrid{9, 1},
                                                   ProductGrid{9, 9}, ProductGrid{1, 9}));
    } // namespace Test
} // namespace SushiBLAS
