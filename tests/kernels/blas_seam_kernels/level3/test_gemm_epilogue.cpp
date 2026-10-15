/**************************************************************************/
/* test_gemm_epilogue.cpp                                                 */
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
 * @file test_gemm_epilogue.cpp
 * @brief Tests that the identity epilogue returns the product element unchanged.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_GemmEpilogue, TheIdentityReturnsItsValueAtEveryPosition)
        {
            const Kernels::GemmIdentityEpilogue identity;

            EXPECT_EQ(identity(2.5f, 0, 0), 2.5f);
            EXPECT_EQ(identity(-7.0, 9, 3), -7.0);
            EXPECT_EQ(identity(std::complex<float>(1.0f, 2.0f), 4, 4),
                      std::complex<float>(1.0f, 2.0f));
        }
    } // namespace Test
} // namespace SushiBLAS
