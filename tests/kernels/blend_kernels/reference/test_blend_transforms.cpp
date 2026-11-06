/**************************************************************************/
/* test_blend_transforms.cpp                                              */
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

/**
 * @file test_blend_transforms.cpp
 * @brief Tests the reference blend of transforms against hand values and a direct loop.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blend_kernels/reference/blend_transforms.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The 3x4 transform that translates by (1, 2, 3) and rotates nothing.
            const std::vector<double> K_TRANSLATE_123{1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3};
        } // namespace

        TEST(Small_ReferenceBlendTransforms, ATranslationMovesAPositionThroughItsFourthColumn)
        {
            const std::vector<double> points{1, 1, 1};
            const std::vector<int32_t> indices{0};
            const std::vector<double> weights{1};
            std::vector<double> out(3, 0.0);

            Kernels::Reference::blend_transforms(1, 3, 3, 1, 1, points.data(),
                                                 K_TRANSLATE_123.data(), indices.data(),
                                                 weights.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{2, 3, 4}));
        }

        TEST(Small_ReferenceBlendTransforms, AHomogeneousDirectionIgnoresTheTranslation)
        {
            const std::vector<double> points{1, 2, 3, 0, 1, 2, 3, 1};
            const std::vector<int32_t> indices{0, 0};
            const std::vector<double> weights{1, 1};
            std::vector<double> out(8, 0.0);

            Kernels::Reference::blend_transforms(2, 4, 3, 1, 1, points.data(),
                                                 K_TRANSLATE_123.data(), indices.data(),
                                                 weights.data(), out.data());

            // The implicit last row carries w through, so a w of 0 stays 0 and a w of 1 stays 1.
            EXPECT_EQ(out, (std::vector<double>{1, 2, 3, 0, 2, 4, 6, 1}));
        }

        TEST(Small_ReferenceBlendTransforms, AFourByFourTransformScalesW)
        {
            const std::vector<double> matrix{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2};
            const std::vector<double> points{1, 2, 3, 5};
            const std::vector<int64_t> indices{0};
            const std::vector<double> weights{1};
            std::vector<double> out(4, 0.0);

            Kernels::Reference::blend_transforms(1, 4, 4, 1, 1, points.data(), matrix.data(),
                                                 indices.data(), weights.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{1, 2, 3, 10}));
        }

        TEST(Small_ReferenceBlendTransforms, AFourByFourTransformOfAPositionDoesNotDivideByW)
        {
            const std::vector<double> matrix{1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 2};
            const std::vector<double> points{1, 1, 1};
            const std::vector<int32_t> indices{0};
            const std::vector<double> weights{1};
            std::vector<double> out(3, 0.0);

            Kernels::Reference::blend_transforms(1, 3, 4, 1, 1, points.data(), matrix.data(),
                                                 indices.data(), weights.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{2, 3, 4}));
        }

        TEST(Small_ReferenceBlendTransforms, TwoInfluencesAverageTheirResults)
        {
            const std::vector<double> matrices{1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0,
                                               1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1, 0};
            const std::vector<double> points{0, 0, 0};
            const std::vector<int32_t> indices{0, 1};
            const std::vector<double> weights{0.5, 0.5};
            std::vector<double> out(3, 0.0);

            Kernels::Reference::blend_transforms(1, 3, 3, 2, 2, points.data(), matrices.data(),
                                                 indices.data(), weights.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{1, 2, 0}));
        }

        TEST(Small_ReferenceBlendTransforms, AnIndexOutsideTheTransformsAddsNoTerm)
        {
            const std::vector<double> points{1, 1, 1};
            const std::vector<int32_t> indices{-1, 0, 1, 2147483647};
            const std::vector<double> weights{1, 1, 1, 1};
            std::vector<double> out(3, 7.0);

            Kernels::Reference::blend_transforms(1, 3, 3, 4, 1, points.data(),
                                                 K_TRANSLATE_123.data(), indices.data(),
                                                 weights.data(), out.data());

            // Only the term of index 0 counts; the other three are outside [0, 1).
            EXPECT_EQ(out, (std::vector<double>{2, 3, 4}));
        }

        /// @brief Holds the component and row counts of one blend the direct loop checks.
        struct ShapeRow
        {
                const char* name;
                int64_t components; ///< 3 for a position, 4 for a homogeneous point.
                int64_t rows;       ///< 3 for an implied last row, 4 for a full transform.
        };

        /// @brief Returns every combination of the two component counts and the two row counts.
        static std::vector<ShapeRow> shape_rows()
        {
            return {{"Position_ThreeRows", 3, 3},
                    {"Position_FourRows", 3, 4},
                    {"Homogeneous_ThreeRows", 4, 3},
                    {"Homogeneous_FourRows", 4, 4}};
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string shape_row_name(const ::testing::TestParamInfo<ShapeRow>& info)
        {
            return info.param.name;
        }

        /**
         * @brief Returns the blend one point at a time, term by term, in the order of the influences.
         *
         * A position has w = 1; a transform of three rows ends in (0, 0, 0, 1); an index outside
         * the transforms adds no term.
         */
        static std::vector<double> direct_blend(const ShapeRow& shape, int64_t point_count,
                                                int64_t influences, int64_t transform_count,
                                                const std::vector<double>& points,
                                                const std::vector<double>& matrices,
                                                const std::vector<int64_t>& indices,
                                                const std::vector<double>& weights)
        {
            std::vector<double> expected(static_cast<std::size_t>(point_count * shape.components), 0.0);
            for (int64_t p = 0; p < point_count; ++p)
            {
                for (int64_t j = 0; j < influences; ++j)
                {
                    const std::size_t at = static_cast<std::size_t>(p * influences + j);
                    const int64_t slot = indices[at];
                    if (slot < 0 || slot >= transform_count)
                        continue;

                    for (int64_t i = 0; i < shape.components; ++i)
                    {
                        double moved = 0.0;
                        for (int64_t c = 0; c < 4; ++c)
                        {
                            const double v = c < shape.components
                                                 ? points[static_cast<std::size_t>(p * shape.components + c)]
                                                 : 1.0;
                            const double m = i < shape.rows
                                                 ? matrices[static_cast<std::size_t>((slot * shape.rows + i) * 4 + c)]
                                                 : (c == 3 ? 1.0 : 0.0);
                            moved += m * v;
                        }
                        expected[static_cast<std::size_t>(p * shape.components + i)] += weights[at] * moved;
                    }
                }
            }
            return expected;
        }

        /// @brief Runs the blend on small integers of one shape.
        class Small_ReferenceBlendTransformsOnShapes : public ::testing::TestWithParam<ShapeRow>
        {
        };

        TEST_P(Small_ReferenceBlendTransformsOnShapes, MatchesADirectLoopOnSmallIntegers)
        {
            constexpr int64_t K_POINTS = 5;
            constexpr int64_t K_TRANSFORMS = 3;
            constexpr int64_t K_INFLUENCES = 4;
            constexpr int64_t K_INDEX_PERIOD = 5;
            const ShapeRow& shape = GetParam();
            const std::vector<double> points =
                small_integers<double>(static_cast<std::size_t>(K_POINTS * shape.components), 1);
            const std::vector<double> matrices = small_integers<double>(
                static_cast<std::size_t>(K_TRANSFORMS * shape.rows * 4), 2);
            const std::vector<double> weights =
                small_integers<double>(static_cast<std::size_t>(K_POINTS * K_INFLUENCES), 3);
            std::vector<int64_t> indices(static_cast<std::size_t>(K_POINTS * K_INFLUENCES));
            std::generate(indices.begin(), indices.end(), [n = int64_t{0}]() mutable
                          { return (n++ % K_INDEX_PERIOD) - 1; });
            std::vector<double> out(static_cast<std::size_t>(K_POINTS * shape.components), 0.0);

            Kernels::Reference::blend_transforms(K_POINTS, shape.components, shape.rows, K_INFLUENCES,
                                                 K_TRANSFORMS, points.data(), matrices.data(),
                                                 indices.data(), weights.data(), out.data());

            EXPECT_EQ(out, direct_blend(shape, K_POINTS, K_INFLUENCES, K_TRANSFORMS, points,
                                        matrices, indices, weights));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceBlendTransformsOnShapes,
                                 ::testing::ValuesIn(shape_rows()), shape_row_name);
    } // namespace Test
} // namespace SushiBLAS
