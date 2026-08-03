/**************************************************************************/
/* test_regression.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// Sentinels for previously fixed bugs — one fast, deterministic test per defect
// so a future change cannot silently reintroduce it.

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Regression_Sentinel : public SushiBLASTest {};

        // SYRK's old oneMKL-backed implementation threw for sycl::half (oneMKL has
        // no half-precision SYRK). The SB-2 tiling-kernel port lifted that
        // restriction, but test_syrk's own TYPED_TEST suite only covers
        // float/double/complex, so nothing would have caught a regression. This is
        // that sentinel.
        TEST_F(Regression_Sentinel, SyrkAcceptsHalfDtype)
        {
            auto A = engine->create_tensor({4, 3}, Core::DataType::HALF);
            auto C = engine->create_tensor({4, 4}, Core::DataType::HALF);

            fill_random<sycl::half>(A);
            fill_random<sycl::half>(C);

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            EXPECT_NO_THROW({
                engine->blas().syrk(A, C, /*upper=*/false, /*transA=*/false, 1.0, 0.0);
                engine->execute().wait();
            }) << "SYRK must accept HALF (SB-2 removed the oneMKL-era restriction).";
        }

        // GEMM and SYRK used to read a COLUMN_MAJOR A and B with row-major
        // indexing - `a[row*lda + col]` where the standard says `a[row + col*lda]`.
        // Two consequences. The obvious one: `layout` was a no-op for both
        // operands, so a column-major GEMM computed the wrong product. The
        // dangerous one: the driver passes the column-major leading dimension
        // (lda = m for an m x k A), so the wrong index runs past the end of the
        // buffer whenever m > k - the maximum touched index is (m-1)*m + k-1
        // against an m*k allocation.
        //
        // It survived because the reference kernel carried the identical defect,
        // so every differential SYCL-vs-reference test agreed. These are the
        // absolute pins that cannot: hand-computed numbers, both execution modes,
        // and a shape with m > k so an out-of-bounds read is back in range of the
        // sanitizers if it ever returns.
        //
        // A (3x2) = [[1,2],[3,4],[5,6]] column-major is [1,3,5,2,4,6], lda = 3.
        // B (2x2) = [[1,0],[1,1]]       column-major is [1,1,0,1],     ldb = 2.
        // A*B     = [[3,2],[7,4],[11,6]] -> column-major [3,7,11,2,4,6], ldc = 3.
        TEST_F(Regression_Sentinel, GemmColumnMajorOperandsUseColumnStrides)
        {
            reinit_engine(Core::Layout::COLUMN_MAJOR);

            const std::vector<float> expected{3, 7, 11, 2, 4, 6};

            for (auto mode : {ExecutionMode::FORCE_SYCL, ExecutionMode::FORCE_REFERENCE})
            {
                SCOPED_TRACE(mode == ExecutionMode::FORCE_SYCL
                    ? "FORCE_SYCL: column-major GEMM must index A and B by column stride"
                    : "FORCE_REFERENCE: column-major GEMM must index A and B by column stride");

                auto A = create_initialized_tensor<float>({3, 2}, {1, 3, 5, 2, 4, 6});
                auto B = create_initialized_tensor<float>({2, 2}, {1, 1, 0, 1});
                auto C = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);

                engine->set_mode(mode);
                engine->blas().gemm(A, B, C);
                engine->execute().wait();

                EXPECT_TENSOR_NEAR(C, expected, 1e-5);
            }
        }

    } // namespace Test
} // namespace SushiBLAS
