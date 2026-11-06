/**************************************************************************/
/* test_level1.cpp                                                        */
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
 * @file test_level1.cpp
 * @brief Tests the reference sparse-dense level 1 kernels against hand-computed results.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>
#include <SushiBLAS/sparse_kernels/reference/level1.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Ref = Kernels::Reference;

        /// @brief The positions 3, 0 and 4 of a dense vector of five, stored zero-based.
        const std::vector<int64_t> K_POSITIONS{3, 0, 4};

        /** @brief Returns the index list of K_POSITIONS. */
        static Kernels::IndexList<int64_t> positions()
        {
            return Kernels::IndexList<int64_t>{3, 5, 0, K_POSITIONS.data()};
        }

        TEST(Small_ReferenceSparseLevel1, AxpyiAddsIntoTheNamedPositions)
        {
            const std::vector<double> x{1.0, 2.0, 3.0};
            std::vector<double> y{10.0, 10.0, 10.0, 10.0, 10.0};

            Ref::axpyi(2.0, x.data(), positions(), y.data(), 1);

            EXPECT_EQ(y, (std::vector<double>{14.0, 10.0, 10.0, 12.0, 16.0}));
        }

        namespace
        {
            /** @brief Returns the complex stored values the doti tests multiply. */
            std::vector<std::complex<double>> doti_values()
            {
                return {{1.0, 1.0}, {0.0, 2.0}, {1.0, 0.0}};
            }
        } // namespace

        TEST(Small_ReferenceSparseLevel1, DotiSumsTheProducts)
        {
            const std::vector<std::complex<double>> x = doti_values();
            const std::vector<std::complex<double>> y(5, std::complex<double>(1.0, 0.0));
            std::complex<double> plain;

            Ref::doti(x.data(), positions(), y.data(), 1, &plain);

            EXPECT_EQ(plain, std::complex<double>(2.0, 3.0));
        }

        TEST(Small_ReferenceSparseLevel1, DotiConjugatesTheStoredValuesWhenAsked)
        {
            const std::vector<std::complex<double>> x = doti_values();
            const std::vector<std::complex<double>> y(5, std::complex<double>(1.0, 0.0));
            std::complex<double> conjugated;

            Ref::doti<std::complex<double>, int64_t, Kernels::Conjugated>(
                x.data(), positions(), y.data(), 1, &conjugated);

            EXPECT_EQ(conjugated, std::complex<double>(2.0, -3.0));
        }

        TEST(Small_ReferenceSparseLevel1, GthrCopiesTheNamedPositionsIntoTheStoredValues)
        {
            const std::vector<double> y{0.0, 1.0, 2.0, 3.0, 4.0};
            std::vector<double> x(3, 0.0);

            Ref::gthr(y.data(), 1, positions(), x.data());

            EXPECT_EQ(x, (std::vector<double>{3.0, 0.0, 4.0}));
        }

        TEST(Small_ReferenceSparseLevel1, GthrzCopiesTheNamedPositionsAndZeroesThem)
        {
            std::vector<double> y{0.0, 1.0, 2.0, 3.0, 4.0};
            std::vector<double> x(3, -1.0);

            Ref::gthrz(y.data(), 1, positions(), x.data());

            EXPECT_EQ(x, (std::vector<double>{3.0, 0.0, 4.0}));
            EXPECT_EQ(y, (std::vector<double>{0.0, 1.0, 2.0, 0.0, 0.0}));
        }

        TEST(Small_ReferenceSparseLevel1, SctrWritesTheStoredValuesIntoTheNamedPositions)
        {
            const std::vector<double> x{7.0, 8.0, 9.0};
            std::vector<double> y(5, -1.0);

            Ref::sctr(x.data(), positions(), y.data(), 1);

            EXPECT_EQ(y, (std::vector<double>{8.0, -1.0, -1.0, 7.0, 9.0}));
        }

        TEST(Small_ReferenceSparseLevel1, RotiRotatesTheStoredPairs)
        {
            std::vector<double> x{1.0, 2.0, 3.0};
            std::vector<double> y{10.0, 20.0, 30.0, 40.0, 50.0};

            Ref::roti(x.data(), positions(), y.data(), 1, 2.0, 1.0);

            // x' = 2x + y and y' = 2y - x on the pairs (1, 40), (2, 10) and (3, 50).
            EXPECT_EQ(x, (std::vector<double>{42.0, 14.0, 56.0}));
            EXPECT_EQ(y, (std::vector<double>{18.0, 20.0, 30.0, 79.0, 97.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
