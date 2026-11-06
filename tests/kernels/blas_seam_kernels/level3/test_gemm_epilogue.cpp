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
        TEST(Small_GemmEpilogue, TheIdentityReturnsItsValueAtEveryPosition)
        {
            const Kernels::GemmIdentityEpilogue identity;

            EXPECT_EQ(identity(2.5f, 0, 0), 2.5f);
            EXPECT_EQ(identity(-7.0, 9, 3), -7.0);
            EXPECT_EQ(identity(std::complex<float>(1.0f, 2.0f), 4, 4),
                      std::complex<float>(1.0f, 2.0f));
        }
    } // namespace Test
} // namespace SushiBLAS
