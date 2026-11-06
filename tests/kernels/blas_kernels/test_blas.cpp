/**************************************************************************/
/* test_blas.cpp                                                          */
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
 * @file test_blas.cpp
 * @brief Tests that blas.hpp alone reaches a reference and a SYCL kernel of each BLAS level.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas_kernels/blas.hpp>

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_BlasKernels, TheReferenceKernelsOfEachLevelAreReachable)
        {
            std::vector<double> y{1.0, 1.0};
            const std::vector<double> x{2.0, 3.0};
            Kernels::Reference::axpy<double>(2, 1.0, x.data(), 1, y.data(), 1);

            const std::vector<double> a{1.0, 0.0, 0.0, 1.0};
            std::vector<double> out{0.0, 0.0};
            Kernels::Reference::gemv(Core::Layout::RowMajor, Core::Transpose::NoTrans, 2, 2, 1.0,
                                     a.data(), 2, y.data(), 1, 0.0, out.data(), 1);

            EXPECT_EQ(out, (std::vector<double>{3.0, 4.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
