/**************************************************************************/
/* test_conjugation.cpp                                                   */
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
 * @file test_conjugation.cpp
 * @brief Tests the two conjugation policies and the types they choose for a kernel.
 * @author Mustafa Garip
 */

#include <complex>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_Conjugation, UnconjugatedLeavesEveryValue)
        {
            const std::complex<double> z(1.0, -2.0);

            EXPECT_EQ(Kernels::Unconjugated::apply(z), z);
            EXPECT_EQ(Kernels::Unconjugated::diagonal(z), z);
            EXPECT_EQ(Kernels::Unconjugated::apply(3.0f), 3.0f);
        }

        TEST(Small_Conjugation, ConjugatedFlipsTheImaginaryPartOfAnElement)
        {
            const std::complex<double> z(1.0, -2.0);

            EXPECT_EQ(Kernels::Conjugated::apply(z), std::complex<double>(1.0, 2.0));
            EXPECT_EQ(Kernels::Conjugated::apply(-4.0), -4.0);
        }

        TEST(Small_Conjugation, ConjugatedRealisesTheDiagonal)
        {
            const std::complex<double> z(1.0, -2.0);

            EXPECT_EQ(Kernels::Conjugated::diagonal(z), std::complex<double>(1.0, 0.0));
        }

        TEST(Small_Conjugation, ARealTypeNeverConjugates)
        {
            const bool real_ignores_policy = std::is_same<Kernels::ConjugationFor<float, Kernels::Conjugated>,
                                                          Kernels::Unconjugated>::value;
            const bool complex_keeps_policy = std::is_same<Kernels::ConjugationFor<std::complex<float>,
                                                                                   Kernels::Conjugated>,
                                                           Kernels::Conjugated>::value;

            EXPECT_TRUE(real_ignores_policy);
            EXPECT_TRUE(complex_keeps_policy);
        }

        TEST(Small_Conjugation, AHermitianUpdateTakesARealScalar)
        {
            const bool her_takes_real = std::is_same<Kernels::RankUpdateScalar<std::complex<double>,
                                                                               Kernels::Conjugated>,
                                                     double>::value;
            const bool syr_takes_element = std::is_same<Kernels::RankUpdateScalar<std::complex<double>,
                                                                                  Kernels::Unconjugated>,
                                                        std::complex<double>>::value;

            EXPECT_TRUE(her_takes_real);
            EXPECT_TRUE(syr_takes_element);
        }
    } // namespace Test
} // namespace SushiBLAS
