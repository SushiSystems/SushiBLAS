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

    } // namespace Test
} // namespace SushiBLAS
