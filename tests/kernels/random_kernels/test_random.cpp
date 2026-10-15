/**************************************************************************/
/* test_random.cpp                                                        */
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
 * @file test_random.cpp
 * @brief Tests that random.hpp alone reaches the draws and both orthonormalisation sweeps.
 * @author Mustafa Garip
 */

#include <SushiBLAS/random_kernels/random.hpp>

#include <array>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_RandomKernels, TheDrawsAndTheReferenceSweepAreReachable)
        {
            using Block = std::array<uint32_t, 4>;
            std::vector<double> p{3.0, 4.0};

            Kernels::Reference::Random::orthonormalize(p.data(), Kernels::VectorSet{1, 2, 1, 2},
                                                       1.0);

            EXPECT_EQ(Kernels::SYCL::Random::philox4x32_10(0, 0, 0, 0, 0, 0),
                      (Block{0x6627E8D5u, 0xE169C58Du, 0xBC57AC4Cu, 0x9B00DBD8u}));
            EXPECT_DOUBLE_EQ(p[0], 0.6);
            EXPECT_DOUBLE_EQ(p[1], 0.8);
            EXPECT_GT(Kernels::SYCL::Random::make_poisson_params(1.0).exp_neg_lambda, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
