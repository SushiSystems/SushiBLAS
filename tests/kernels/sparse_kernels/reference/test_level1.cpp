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

        TEST(Small_ReferenceSparseLevel1, DotiSumsTheProductsAndConjugatesWhenAsked)
        {
            const std::vector<std::complex<double>> x{{1.0, 1.0}, {0.0, 2.0}, {1.0, 0.0}};
            const std::vector<std::complex<double>> y(5, std::complex<double>(1.0, 0.0));
            std::complex<double> plain;
            std::complex<double> conjugated;

            Ref::doti(x.data(), positions(), y.data(), 1, &plain);
            Ref::doti<std::complex<double>, int64_t, Kernels::Conjugated>(
                x.data(), positions(), y.data(), 1, &conjugated);

            EXPECT_EQ(plain, std::complex<double>(2.0, 3.0));
            EXPECT_EQ(conjugated, std::complex<double>(2.0, -3.0));
        }

        TEST(Small_ReferenceSparseLevel1, GatherScatterAndGatherWithZero)
        {
            std::vector<double> y{0.0, 1.0, 2.0, 3.0, 4.0};
            std::vector<double> x(3, 0.0);

            Ref::gthr(y.data(), 1, positions(), x.data());
            EXPECT_EQ(x, (std::vector<double>{3.0, 0.0, 4.0}));

            Ref::gthrz(y.data(), 1, positions(), x.data());
            EXPECT_EQ(x, (std::vector<double>{3.0, 0.0, 4.0}));
            EXPECT_EQ(y, (std::vector<double>{0.0, 1.0, 2.0, 0.0, 0.0}));

            Ref::sctr(x.data(), positions(), y.data(), 1);
            EXPECT_EQ(y, (std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0}));
        }

        TEST(Small_ReferenceSparseLevel1, RotiRotatesTheStoredPairs)
        {
            std::vector<double> x{1.0, 2.0, 3.0};
            std::vector<double> y{10.0, 20.0, 30.0, 40.0, 50.0};

            Ref::roti(x.data(), positions(), y.data(), 1, 0.0, 1.0);

            EXPECT_EQ(x, (std::vector<double>{40.0, 10.0, 50.0}));
            EXPECT_EQ(y, (std::vector<double>{-2.0, 20.0, 30.0, -1.0, -3.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
