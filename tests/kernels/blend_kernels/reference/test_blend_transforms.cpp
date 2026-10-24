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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */

/**
 * @file test_blend_transforms.cpp
 * @brief Tests the reference blend of transforms against hand values and a direct loop.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
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

        TEST(Small_ReferenceBlendTransforms, MatchesADirectLoopOnSmallIntegers)
        {
            constexpr int64_t K_POINTS = 5;
            constexpr int64_t K_TRANSFORMS = 3;
            constexpr int64_t K_INFLUENCES = 4;

            for (const int64_t components : {int64_t{3}, int64_t{4}})
            {
                for (const int64_t rows : {int64_t{3}, int64_t{4}})
                {
                    const std::vector<double> points =
                        small_integers<double>(static_cast<std::size_t>(K_POINTS * components), 1);
                    const std::vector<double> matrices = small_integers<double>(
                        static_cast<std::size_t>(K_TRANSFORMS * rows * 4), 2);
                    const std::vector<double> weights = small_integers<double>(
                        static_cast<std::size_t>(K_POINTS * K_INFLUENCES), 3);
                    std::vector<int64_t> indices(static_cast<std::size_t>(K_POINTS * K_INFLUENCES));
                    for (std::size_t i = 0; i < indices.size(); ++i)
                        indices[i] = static_cast<int64_t>(i % 5) - 1;
                    std::vector<double> out(static_cast<std::size_t>(K_POINTS * components), 0.0);
                    std::vector<double> expected(out.size(), 0.0);

                    Kernels::Reference::blend_transforms(K_POINTS, components, rows, K_INFLUENCES,
                                                         K_TRANSFORMS, points.data(),
                                                         matrices.data(), indices.data(),
                                                         weights.data(), out.data());

                    for (int64_t p = 0; p < K_POINTS; ++p)
                    {
                        for (int64_t j = 0; j < K_INFLUENCES; ++j)
                        {
                            const std::size_t at = static_cast<std::size_t>(p * K_INFLUENCES + j);
                            const int64_t slot = indices[at];
                            if (slot < 0 || slot >= K_TRANSFORMS)
                                continue;

                            for (int64_t i = 0; i < components; ++i)
                            {
                                double moved = 0.0;
                                for (int64_t c = 0; c < 4; ++c)
                                {
                                    // A position has w = 1; a 3-row transform ends in (0, 0, 0, 1).
                                    double v = 1.0;
                                    if (c < components)
                                        v = points[static_cast<std::size_t>(p * components + c)];
                                    double m = c == 3 ? 1.0 : 0.0;
                                    if (i < rows)
                                        m = matrices[static_cast<std::size_t>(
                                            (slot * rows + i) * 4 + c)];
                                    moved += m * v;
                                }
                                expected[static_cast<std::size_t>(p * components + i)] +=
                                    weights[at] * moved;
                            }
                        }
                    }

                    EXPECT_EQ(out, expected) << components << " values, " << rows << " rows";
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
