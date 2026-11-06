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
 * @file test_orthonormalize.cpp
 * @brief Tests that the reference sweep leaves strided vectors orthogonal and of the given norm.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>
#include <SushiBLAS/random_kernels/reference/orthonormalize.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance: 1024 double roundings of a value of order one.
        constexpr double K_TOLERANCE = absolute_tolerance<double>(1024);

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

        /** @brief Expects every Gram entry of the set to be zero off the diagonal and diagonal on it. */
        static void expect_orthogonal_of_norm(const std::vector<double>& p,
                                              const Kernels::VectorSet& set, double diagonal)
        {
            for (int64_t i = 0; i < set.count; ++i)
                for (int64_t j = 0; j < set.count; ++j)
                    EXPECT_NEAR(dot_of(p, set, i, j), (i == j) ? diagonal : 0.0, K_TOLERANCE)
                        << i << ", " << j;
        }

        /// @brief Holds one layout of vectors, the integers they start from and the gain.
        struct SweepRow
        {
                const char* name;
                Kernels::VectorSet set;
                uint32_t seed;                    ///< Seed of the small integers the vectors start from.
                std::vector<std::size_t> boosted; ///< Elements raised so the set is not rank deficient.
                double boost;                     ///< What each boosted element gains.
                double gain;                      ///< The norm every vector is scaled to.
        };

        /// @brief Returns a row-major layout at gain 2 and a column-major layout at gain 1.
        static std::vector<SweepRow> sweep_rows()
        {
            return {{"RowsAtGainTwo", Kernels::VectorSet{4, 7, 1, 7}, 1, {0}, 10.0, 2.0},
                    {"ColumnsOfAColumnMajorMatrixAtGainOne", Kernels::VectorSet{4, 6, 1, 6}, 2,
                     {0, 7, 14, 21}, 8.0, 1.0}};
        }

        /// @brief Returns the small integers of a row with its boosted elements raised.
        static std::vector<double> starting_vectors(const SweepRow& row)
        {
            std::vector<double> p = small_integers<double>(
                static_cast<std::size_t>(row.set.count * row.set.length), row.seed);
            for (const std::size_t at : row.boosted)
                p[at] += row.boost;
            return p;
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string sweep_row_name(const ::testing::TestParamInfo<SweepRow>& info)
        {
            return info.param.name;
        }

        /// @brief Runs the sweep on one layout.
        class Small_ReferenceOrthonormalizeLayouts : public ::testing::TestWithParam<SweepRow>
        {
        };

        TEST_P(Small_ReferenceOrthonormalizeLayouts, VectorsBecomeOrthogonalAtTheGain)
        {
            const SweepRow& row = GetParam();
            std::vector<double> p = starting_vectors(row);

            Kernels::Reference::Random::orthonormalize(p.data(), row.set, row.gain);

            expect_orthogonal_of_norm(p, row.set, row.gain * row.gain);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceOrthonormalizeLayouts,
                                 ::testing::ValuesIn(sweep_rows()), sweep_row_name);

        TEST(Small_ReferenceOrthonormalize, AZeroVectorStaysZero)
        {
            const Kernels::VectorSet set{2, 3, 1, 3};
            std::vector<double> p{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            Kernels::Reference::Random::orthonormalize(p.data(), set, 1.0);

            EXPECT_EQ(p, (std::vector<double>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
