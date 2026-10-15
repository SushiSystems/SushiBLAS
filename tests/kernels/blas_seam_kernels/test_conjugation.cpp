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

        TEST(Small_Conjugation, ConjugatedFlipsTheImaginaryPartAndRealisesTheDiagonal)
        {
            const std::complex<double> z(1.0, -2.0);

            EXPECT_EQ(Kernels::Conjugated::apply(z), std::complex<double>(1.0, 2.0));
            EXPECT_EQ(Kernels::Conjugated::diagonal(z), std::complex<double>(1.0, 0.0));
            EXPECT_EQ(Kernels::Conjugated::apply(-4.0), -4.0);
        }

        TEST(Small_Conjugation, ARealTypeNeverConjugates)
        {
            static_assert(std::is_same<Kernels::ConjugationFor<float, Kernels::Conjugated>,
                                       Kernels::Unconjugated>::value,
                          "A real type ignores the conjugating policy.");
            static_assert(std::is_same<Kernels::ConjugationFor<std::complex<float>,
                                                               Kernels::Conjugated>,
                                       Kernels::Conjugated>::value,
                          "A complex type keeps it.");

            SUCCEED();
        }

        TEST(Small_Conjugation, AHermitianUpdateTakesARealScalar)
        {
            static_assert(std::is_same<Kernels::RankUpdateScalar<std::complex<double>,
                                                                 Kernels::Conjugated>,
                                       double>::value,
                          "HER takes a real alpha.");
            static_assert(std::is_same<Kernels::RankUpdateScalar<std::complex<double>,
                                                                 Kernels::Unconjugated>,
                                       std::complex<double>>::value,
                          "SYR takes an alpha of the element type.");

            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
