/**************************************************************************/
/* test_orthonormalize.cpp                                                */
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
 * @file test_orthonormalize.cpp
 * @brief Tests that the reference sweep leaves strided vectors orthogonal and of the given norm.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>
#include <SushiBLAS/random_kernels/reference/orthonormalize.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance of an inner product the sweep must zero or set.
        constexpr double K_TOLERANCE = 1e-12;

        /** @brief Returns the dot product of vectors i and j of a set. */
        static double dot_of(const std::vector<double>& p, const Kernels::VectorSet& set,
                             int64_t i, int64_t j)
        {
            double sum = 0.0;
            for (int64_t k = 0; k < set.length; ++k)
                sum += p[static_cast<std::size_t>(i * set.vec_stride + k * set.elem_stride)] *
                       p[static_cast<std::size_t>(j * set.vec_stride + k * set.elem_stride)];
            return sum;
        }

        TEST(Unit_ReferenceOrthonormalize, RowsBecomeOrthonormalAtTheGain)
        {
            const Kernels::VectorSet set{4, 7, 1, 7};
            std::vector<double> p = small_integers<double>(28, 1);
            p[0] += 10.0;

            Kernels::Reference::Random::orthonormalize(p.data(), set, 2.0);

            for (int64_t i = 0; i < set.count; ++i)
                for (int64_t j = 0; j < set.count; ++j)
                    EXPECT_NEAR(dot_of(p, set, i, j), (i == j) ? 4.0 : 0.0, K_TOLERANCE)
                        << i << ", " << j;
        }

        TEST(Unit_ReferenceOrthonormalize, ColumnsOfAColumnMajorMatrixWorkThroughTheStrides)
        {
            // Four column vectors of length six stored column by column in a 6 x 4 matrix.
            const Kernels::VectorSet set{4, 6, 1, 6};
            std::vector<double> p = small_integers<double>(24, 2);
            for (int64_t v = 0; v < 4; ++v)
                p[static_cast<std::size_t>(v * 6 + v)] += 8.0;

            Kernels::Reference::Random::orthonormalize(p.data(), set, 1.0);

            for (int64_t i = 0; i < set.count; ++i)
                for (int64_t j = 0; j < set.count; ++j)
                    EXPECT_NEAR(dot_of(p, set, i, j), (i == j) ? 1.0 : 0.0, K_TOLERANCE);
        }

        TEST(Unit_ReferenceOrthonormalize, AZeroVectorStaysZero)
        {
            const Kernels::VectorSet set{2, 3, 1, 3};
            std::vector<double> p{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            Kernels::Reference::Random::orthonormalize(p.data(), set, 1.0);

            EXPECT_EQ(p, (std::vector<double>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
