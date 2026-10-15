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
 * @brief Tests the reference Level 1 kernels against hand-computed results and strides.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/level1.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Ref = Kernels::Reference;

        TEST(Unit_ReferenceLevel1, AxpyCopyScalAndSwapFollowTheirStrides)
        {
            const std::vector<double> x{1.0, 9.0, 2.0, 9.0, 3.0};
            std::vector<double> y{10.0, 20.0, 30.0};

            Ref::axpy<double>(3, 2.0, x.data(), 2, y.data(), 1);
            EXPECT_EQ(y, (std::vector<double>{12.0, 24.0, 36.0}));

            Ref::scal<double>(3, -1.0, y.data(), 1);
            EXPECT_EQ(y, (std::vector<double>{-12.0, -24.0, -36.0}));

            std::vector<double> copied(3, 0.0);
            Ref::copy<double>(3, x.data(), 2, copied.data(), 1);
            EXPECT_EQ(copied, (std::vector<double>{1.0, 2.0, 3.0}));

            Ref::swap<double>(3, copied.data(), 1, y.data(), 1);
            EXPECT_EQ(y, (std::vector<double>{1.0, 2.0, 3.0}));
            EXPECT_EQ(copied, (std::vector<double>{-12.0, -24.0, -36.0}));
        }

        TEST(Unit_ReferenceLevel1, RotTurnsEachPairByCosineAndSine)
        {
            std::vector<double> x{1.0, 2.0};
            std::vector<double> y{3.0, 4.0};

            Ref::rot<double>(2, x.data(), 1, y.data(), 1, 0.0, 1.0);

            EXPECT_EQ(x, (std::vector<double>{3.0, 4.0}));
            EXPECT_EQ(y, (std::vector<double>{-1.0, -2.0}));
        }

        TEST(Unit_ReferenceLevel1, AsumAddsMagnitudesAndBothPartsOfAComplex)
        {
            const std::vector<float> real{1.0f, -2.0f, 3.0f};
            const std::vector<std::complex<float>> complex{{1.0f, -2.0f}, {-3.0f, 4.0f}};
            float real_sum = 0.0f;
            float complex_sum = 0.0f;

            Ref::asum<float>(3, real.data(), 1, &real_sum);
            Ref::asum<std::complex<float>>(2, complex.data(), 1, &complex_sum);

            EXPECT_EQ(real_sum, 6.0f);
            EXPECT_EQ(complex_sum, 10.0f);
        }

        TEST(Unit_ReferenceLevel1, Nrm2IsTheEuclideanNormWithoutOverflow)
        {
            const std::vector<double> small{3.0, 4.0};
            const std::vector<float> large{3.0e30f, 4.0e30f};
            double small_norm = 0.0;
            float large_norm = 0.0f;

            Ref::nrm2<double>(2, small.data(), 1, &small_norm);
            Ref::nrm2<float>(2, large.data(), 1, &large_norm);

            EXPECT_EQ(small_norm, 5.0);
            EXPECT_FLOAT_EQ(large_norm, 5.0e30f);
        }

        TEST(Unit_ReferenceLevel1, DotSumsTheProductsAndConjugatesWhenAsked)
        {
            const std::vector<std::complex<double>> x{{1.0, 1.0}, {2.0, 0.0}};
            const std::vector<std::complex<double>> y{{3.0, 0.0}, {0.0, 1.0}};
            std::complex<double> plain;
            std::complex<double> conjugated;

            Ref::dot<std::complex<double>>(2, x.data(), 1, y.data(), 1, &plain);
            Ref::dot<std::complex<double>, Kernels::Conjugated>(2, x.data(), 1, y.data(), 1,
                                                                &conjugated);

            EXPECT_EQ(plain, std::complex<double>(3.0, 5.0));
            EXPECT_EQ(conjugated, std::complex<double>(3.0, -1.0));
        }

        TEST(Unit_ReferenceLevel1, IamaxReturnsTheFirstLargestMagnitude)
        {
            const std::vector<double> x{1.0, -7.0, 7.0, 2.0};
            const std::vector<std::complex<float>> z{{1.0f, 1.0f}, {0.0f, -3.0f}, {2.0f, 0.5f}};
            int64_t real_index = -1;
            int64_t complex_index = -1;

            Ref::iamax<double>(4, x.data(), 1, &real_index);
            Ref::iamax<std::complex<float>>(3, z.data(), 1, &complex_index);

            EXPECT_EQ(real_index, int64_t{1});
            EXPECT_EQ(complex_index, int64_t{1});
        }

        TEST(Unit_ReferenceLevel1, AbsOneAddsTheMagnitudesOfBothParts)
        {
            EXPECT_EQ(Ref::abs1(std::complex<double>(-1.5, 2.0)), 3.5);
            EXPECT_EQ(Ref::abs1(-4.0f), 4.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
