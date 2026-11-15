/**************************************************************************/
/* test_gemm_split.cpp                                                    */
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
 * @file test_gemm_split.cpp
 * @brief Tests the axis, the sub-problem and the slice kernel of a GEMM split.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/gemm_split.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Core::Layout;
        using Core::Transpose;
        using Dispatch::GemmGeometry;
        using Dispatch::GemmSplitAxis;

        namespace
        {
            /** @brief Adds the position of an element to its value, so a wrong offset shows. */
            struct PositionEpilogue
            {
                /** @brief Returns value plus 100 per row and 1 per column. */
                float operator()(float value, int64_t row, int64_t column) const noexcept
                {
                    return value + static_cast<float>(100 * row + column);
                }
            };

            /** @brief Holds one product: its layout, its flags and its extents. */
            struct SplitCase
            {
                Layout layout;      /**< The layout of A, B and C. */
                Transpose trans_a;  /**< How A is read. */
                Transpose trans_b;  /**< How B is read. */
                int64_t m;          /**< Rows of C. */
                int64_t n;          /**< Columns of C. */
                int64_t k;          /**< The summed extent. */
            };

            /** @brief Returns the geometry of a contiguous, unbatched product. */
            [[nodiscard]] GemmGeometry geometry_of(const SplitCase& test_case)
            {
                const bool row_major = test_case.layout == Layout::RowMajor;
                const bool a_swapped = Core::is_transposed(test_case.trans_a);
                const bool b_swapped = Core::is_transposed(test_case.trans_b);
                GemmGeometry geometry;
                geometry.m = test_case.m;
                geometry.n = test_case.n;
                geometry.k = test_case.k;
                geometry.a.leading_dimension = row_major ? (a_swapped ? test_case.m : test_case.k)
                                                         : (a_swapped ? test_case.k : test_case.m);
                geometry.b.leading_dimension = row_major ? (b_swapped ? test_case.k : test_case.n)
                                                         : (b_swapped ? test_case.n : test_case.k);
                geometry.c.leading_dimension = row_major ? test_case.n : test_case.m;
                return geometry;
            }

            /** @brief Returns count small integers that differ from one element to the next. */
            [[nodiscard]] std::vector<float> values(std::size_t count, int seed)
            {
                std::vector<float> result(count);
                for (std::size_t i = 0; i < count; ++i)
                    result[i] = static_cast<float>((static_cast<int>(i) * 7 + seed) % 11 - 5);
                return result;
            }
        } // namespace

        TEST(Small_GemmSplitAxis, TheLongerSideOfCIsCut)
        {
            GemmGeometry tall;
            tall.m = 8;
            tall.n = 3;
            GemmGeometry wide;
            wide.m = 3;
            wide.n = 8;

            EXPECT_EQ(Dispatch::gemm_split_axis(tall), GemmSplitAxis::Rows);
            EXPECT_EQ(Dispatch::gemm_split_axis(wide), GemmSplitAxis::Columns);
        }

        TEST(Small_GemmSplitAxis, ASquareCIsCutAlongItsRows)
        {
            GemmGeometry square;
            square.m = 4;
            square.n = 4;

            EXPECT_EQ(Dispatch::gemm_split_axis(square), GemmSplitAxis::Rows);
        }

        TEST(Small_GemmSplitAxis, TheExtentIsTheLengthOfTheAxis)
        {
            GemmGeometry geometry;
            geometry.m = 8;
            geometry.n = 3;

            EXPECT_EQ(Dispatch::gemm_split_extent(geometry, GemmSplitAxis::Rows), int64_t{8});
            EXPECT_EQ(Dispatch::gemm_split_extent(geometry, GemmSplitAxis::Columns), int64_t{3});
        }

        TEST(Small_GemmSlice, ARowPartOfARowMajorProductStepsAByItsLeadingDimension)
        {
            const SplitCase test_case{Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                                      6, 4, 3};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Rows, 2, 5);

            EXPECT_EQ(slice.m, int64_t{3});
            EXPECT_EQ(slice.n, int64_t{4});
            EXPECT_EQ(slice.a_offset, int64_t{2 * 3});
            EXPECT_EQ(slice.b_offset, int64_t{0});
            EXPECT_EQ(slice.c_offset, int64_t{2 * 4});
        }

        TEST(Small_GemmSlice, AColumnPartOfARowMajorProductStepsBByOne)
        {
            const SplitCase test_case{Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                                      3, 6, 4};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Columns, 2, 5);

            EXPECT_EQ(slice.m, int64_t{3});
            EXPECT_EQ(slice.n, int64_t{3});
            EXPECT_EQ(slice.a_offset, int64_t{0});
            EXPECT_EQ(slice.b_offset, int64_t{2});
            EXPECT_EQ(slice.c_offset, int64_t{2});
        }

        TEST(Small_GemmSlice, ATransposedAStepsByOneRowMajor)
        {
            const SplitCase test_case{Layout::RowMajor, Transpose::Trans, Transpose::NoTrans,
                                      6, 4, 3};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Rows, 2, 5);

            EXPECT_EQ(slice.a_offset, int64_t{2});
        }

        TEST(Small_GemmSlice, AColumnMajorProductStepsCByOneAlongItsRows)
        {
            const SplitCase test_case{Layout::ColumnMajor, Transpose::NoTrans, Transpose::NoTrans,
                                      6, 4, 3};

            const Dispatch::GemmSlice slice = Dispatch::slice_gemm(
                geometry_of(test_case), test_case.layout, test_case.trans_a, test_case.trans_b,
                GemmSplitAxis::Rows, 2, 5);

            EXPECT_EQ(slice.a_offset, int64_t{2});
            EXPECT_EQ(slice.c_offset, int64_t{2});
        }

        TEST(Small_OffsetEpilogue, TheFunctorSeesThePositionInTheWholeMatrix)
        {
            Dispatch::OffsetEpilogue<PositionEpilogue> epilogue(PositionEpilogue{}, 3, 5);

            EXPECT_EQ(epilogue(1.0f, 2, 4), 1.0f + static_cast<float>(100 * 5 + 9));
        }

        /** @brief Runs a product whole and in parts and compares the two results. */
        class Small_GemmSplitSlice : public ::testing::TestWithParam<SplitCase>
        {
            protected:
                /**
                 * @brief Returns the C that three consecutive parts of the declared split compute.
                 * @param test_case The product to compute.
                 * @param a The elements of A.
                 * @param b The elements of B.
                 */
                [[nodiscard]] static std::vector<float> in_parts(const SplitCase& test_case,
                                                                 const std::vector<float>& a,
                                                                 const std::vector<float>& b)
                {
                    std::vector<float> c(static_cast<std::size_t>(test_case.m * test_case.n), 0.0f);
                    const GemmGeometry geometry = geometry_of(test_case);
                    const auto split = Dispatch::make_gemm_split<float>(
                        geometry, test_case.layout, test_case.trans_a, test_case.trans_b, 1.0f,
                        a.data(), b.data(), 0.0f, c.data(), PositionEpilogue{});

                    const std::uint64_t extent = split->extent;
                    const std::uint64_t first = extent / 3;
                    const std::uint64_t second = extent - extent / 3;
                    const std::uint64_t bounds[4] = {0, first, second, extent};
                    for (std::size_t index = 0; index < 3; ++index)
                    {
                        split->slice(SushiRuntime::SplitPart{split->axis, bounds[index],
                                                             bounds[index + 1], index, 3});
                    }
                    return c;
                }

                /** @brief Returns the C that the whole product computes. */
                [[nodiscard]] static std::vector<float> whole(const SplitCase& test_case,
                                                              const std::vector<float>& a,
                                                              const std::vector<float>& b)
                {
                    std::vector<float> c(static_cast<std::size_t>(test_case.m * test_case.n), 0.0f);
                    const GemmGeometry geometry = geometry_of(test_case);
                    Kernels::Reference::gemm<float>(
                        test_case.layout, test_case.trans_a, test_case.trans_b, test_case.m,
                        test_case.n, test_case.k, 1.0f, a.data(), geometry.a.leading_dimension,
                        b.data(), geometry.b.leading_dimension, 0.0f, c.data(),
                        geometry.c.leading_dimension, PositionEpilogue{});
                    return c;
                }
        };

        TEST_P(Small_GemmSplitSlice, ThePartsTogetherComputeTheWholeProduct)
        {
            const SplitCase test_case = GetParam();
            const std::vector<float> a = values(static_cast<std::size_t>(test_case.m * test_case.k), 1);
            const std::vector<float> b = values(static_cast<std::size_t>(test_case.k * test_case.n), 4);

            EXPECT_EQ(in_parts(test_case, a, b), whole(test_case, a, b));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GemmSplitSlice,
            ::testing::Values(
                SplitCase{Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, 9, 4, 5},
                SplitCase{Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, 4, 9, 5},
                SplitCase{Layout::RowMajor, Transpose::Trans, Transpose::NoTrans, 9, 4, 5},
                SplitCase{Layout::RowMajor, Transpose::NoTrans, Transpose::Trans, 4, 9, 5},
                SplitCase{Layout::ColumnMajor, Transpose::NoTrans, Transpose::NoTrans, 9, 4, 5},
                SplitCase{Layout::ColumnMajor, Transpose::NoTrans, Transpose::NoTrans, 4, 9, 5},
                SplitCase{Layout::ColumnMajor, Transpose::Trans, Transpose::Trans, 9, 4, 5},
                SplitCase{Layout::ColumnMajor, Transpose::Trans, Transpose::Trans, 4, 9, 5}));
    } // namespace Test
} // namespace SushiBLAS
