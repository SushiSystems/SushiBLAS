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

        /// @brief Tests that axpy adds alpha times a strided x to y.
        TEST(Small_ReferenceLevel1, AxpyAddsAlphaTimesAStridedX)
        {
            const std::vector<double> x{1.0, 9.0, 2.0, 9.0, 3.0};
            std::vector<double> y{10.0, 20.0, 30.0};

            Ref::axpy<double>(3, 2.0, x.data(), 2, y.data(), 1);

            EXPECT_EQ(y, (std::vector<double>{12.0, 24.0, 36.0}));
        }

        /// @brief Tests that scal multiplies every element by alpha.
        TEST(Small_ReferenceLevel1, ScalMultipliesEveryElementByAlpha)
        {
            std::vector<double> y{12.0, 24.0, 36.0};

            Ref::scal<double>(3, -1.0, y.data(), 1);

            EXPECT_EQ(y, (std::vector<double>{-12.0, -24.0, -36.0}));
        }

        /// @brief Tests that copy reads a strided source into a contiguous destination.
        TEST(Small_ReferenceLevel1, CopyReadsAStridedSource)
        {
            const std::vector<double> x{1.0, 9.0, 2.0, 9.0, 3.0};
            std::vector<double> copied(3, 0.0);

            Ref::copy<double>(3, x.data(), 2, copied.data(), 1);

            EXPECT_EQ(copied, (std::vector<double>{1.0, 2.0, 3.0}));
        }

        /// @brief Tests that swap exchanges the two vectors.
        TEST(Small_ReferenceLevel1, SwapExchangesTheTwoVectors)
        {
            std::vector<double> x{1.0, 2.0, 3.0};
            std::vector<double> y{-12.0, -24.0, -36.0};

            Ref::swap<double>(3, x.data(), 1, y.data(), 1);

            EXPECT_EQ(y, (std::vector<double>{1.0, 2.0, 3.0}));
            EXPECT_EQ(x, (std::vector<double>{-12.0, -24.0, -36.0}));
        }

        /// @brief Tests that rot turns each pair by the cosine and the sine.
        TEST(Small_ReferenceLevel1, RotTurnsEachPairByCosineAndSine)
        {
            std::vector<double> x{1.0, 2.0};
            std::vector<double> y{3.0, 4.0};

            Ref::rot<double>(2, x.data(), 1, y.data(), 1, 0.0, 1.0);

            EXPECT_EQ(x, (std::vector<double>{3.0, 4.0}));
            EXPECT_EQ(y, (std::vector<double>{-1.0, -2.0}));
        }

        /// @brief Tests that rot with a cosine and a sine that are both nonzero changes both components by the rotation formula.
        TEST(Small_ReferenceLevel1, RotWithBothCosineAndSineNonzeroMixesTheComponents)
        {
            std::vector<double> x{1.0, 2.0};
            std::vector<double> y{3.0, 4.0};

            Ref::rot<double>(2, x.data(), 1, y.data(), 1, 0.6, 0.8);

            EXPECT_DOUBLE_EQ(x[0], 3.0);
            EXPECT_DOUBLE_EQ(x[1], 4.4);
            EXPECT_DOUBLE_EQ(y[0], 1.0);
            EXPECT_DOUBLE_EQ(y[1], 0.8);
        }

        /// @brief Tests that asum of a real vector adds the magnitudes.
        TEST(Small_ReferenceLevel1, AsumAddsTheMagnitudesOfARealVector)
        {
            const std::vector<float> real{1.0f, -2.0f, 3.0f};
            float sum = 0.0f;

            Ref::asum<float>(3, real.data(), 1, &sum);

            EXPECT_EQ(sum, 6.0f);
        }

        /// @brief Tests that asum of a complex vector adds the real and imaginary magnitudes.
        TEST(Small_ReferenceLevel1, AsumAddsBothPartsOfAComplexVector)
        {
            const std::vector<std::complex<float>> complex{{1.0f, -2.0f}, {-3.0f, 4.0f}};
            float sum = 0.0f;

            Ref::asum<std::complex<float>>(2, complex.data(), 1, &sum);

            EXPECT_EQ(sum, 10.0f);
        }

        /// @brief Tests that nrm2 is the Euclidean norm.
        TEST(Small_ReferenceLevel1, Nrm2IsTheEuclideanNorm)
        {
            const std::vector<double> x{3.0, 4.0};
            double norm = 0.0;

            Ref::nrm2<double>(2, x.data(), 1, &norm);

            EXPECT_EQ(norm, 5.0);
        }

        /// @brief Tests that nrm2 does not overflow where the squares exceed the float range.
        TEST(Small_ReferenceLevel1, Nrm2DoesNotOverflowWhereTheSquaresDo)
        {
            const std::vector<float> large{3.0e30f, 4.0e30f};
            float norm = 0.0f;

            Ref::nrm2<float>(2, large.data(), 1, &norm);

            EXPECT_FLOAT_EQ(norm, 5.0e30f);
        }

        /// @brief Tests that dot sums the products without conjugation.
        TEST(Small_ReferenceLevel1, DotSumsTheProducts)
        {
            const std::vector<std::complex<double>> x{{1.0, 1.0}, {2.0, 0.0}};
            const std::vector<std::complex<double>> y{{3.0, 0.0}, {0.0, 1.0}};
            std::complex<double> plain;

            Ref::dot<std::complex<double>>(2, x.data(), 1, y.data(), 1, &plain);

            EXPECT_EQ(plain, std::complex<double>(3.0, 5.0));
        }

        /// @brief Tests that dot conjugates x when asked.
        TEST(Small_ReferenceLevel1, DotConjugatesXWhenAsked)
        {
            const std::vector<std::complex<double>> x{{1.0, 1.0}, {2.0, 0.0}};
            const std::vector<std::complex<double>> y{{3.0, 0.0}, {0.0, 1.0}};
            std::complex<double> conjugated;

            Ref::dot<std::complex<double>, Kernels::Conjugated>(2, x.data(), 1, y.data(), 1,
                                                                &conjugated);

            EXPECT_EQ(conjugated, std::complex<double>(3.0, -1.0));
        }

        /// @brief Tests that iamax of a real vector returns the first largest magnitude.
        TEST(Small_ReferenceLevel1, IamaxOfARealVectorReturnsTheFirstLargestMagnitude)
        {
            const std::vector<double> x{1.0, -7.0, 7.0, 2.0};
            int64_t index = -1;

            Ref::iamax<double>(4, x.data(), 1, &index);

            EXPECT_EQ(index, int64_t{1});
        }

        /// @brief Tests that iamax of a complex vector ranks by abs1, not by modulus.
        TEST(Small_ReferenceLevel1, IamaxOfAComplexVectorRanksByAbsOne)
        {
            // abs1 is 6, 5 and 2; the modulus is 4.24, 5 and 1.41, so only abs1 picks index 0.
            const std::vector<std::complex<float>> z{{3.0f, 3.0f}, {0.0f, -5.0f}, {1.0f, 1.0f}};
            int64_t index = -1;

            Ref::iamax<std::complex<float>>(3, z.data(), 1, &index);

            EXPECT_EQ(index, int64_t{0});
        }

        /// @brief Tests that abs1 of a complex adds the magnitudes of both parts.
        TEST(Small_ReferenceLevel1, AbsOneOfAComplexAddsTheMagnitudesOfBothParts)
        {
            EXPECT_EQ(Ref::abs1(std::complex<double>(-1.5, 2.0)), 3.5);
        }

        /// @brief Tests that abs1 of a real is its magnitude.
        TEST(Small_ReferenceLevel1, AbsOneOfARealIsItsMagnitude)
        {
            EXPECT_EQ(Ref::abs1(-4.0f), 4.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
