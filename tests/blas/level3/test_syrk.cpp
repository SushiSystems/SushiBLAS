/**************************************************************************/
/* test_syrk.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"


class SYRKTest : public SushiBLASTest {};

TEST_F(SYRKTest, SimpleSYRK) 
{
    const int N = 3;
    const int K = 2;
    
    // C is N x N, A is N x K (if transA=false)
    auto A = engine->create_tensor({N, K});
    auto C = engine->create_tensor({N, N});

    // A: 
    // [1 2]
    // [3 4]
    // [5 6]
    fill_tensor(A, {1, 2, 3, 4, 5, 6});
    
    // C: initialized to zeros
    fill_tensor(C, std::vector<float>(N * N, 0.0f));

    SB_LOG_INFO("Submitting standard SYRK.");
    // Upper=false (lower triangular), TransA=false, alpha=1.0, beta=0.0
    engine->blas().syrk(A, C, false, false, 1.0f, 0.0f);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // C = A * A^T
    // [1 2]   [1 3 5]   [ 5 11 17]
    // [3 4] * [2 4 6] = [11 25 39]
    // [5 6]             [17 39 61]
    // Since uplo=lower (false), the upper elements are untouched (remain 0).
    verify_tensor(C, {5, 0, 0, 11, 25, 0, 17, 39, 61});
}

TEST_F(SYRKTest, SimpleSYRKColumnMajor) 
{
    reinit_engine(sb::Core::Layout::COLUMN_MAJOR);
    
    const int N = 2, K = 3;
    auto A = engine->create_tensor({K, N}); // transA=true means A is K x N
    auto C = engine->create_tensor({N, N});

    // A: 
    // [1 4]
    // [2 5]
    // [3 6]
    // Column-major in memory: [1 2 3 4 5 6]
    fill_tensor(A, {1, 2, 3, 4, 5, 6});
    
    // C: initialized to zeros
    fill_tensor(C, std::vector<float>(N * N, 0.0f));

    SB_LOG_INFO("Submitting Column-Major SYRK.");
    // Upper=false, TransA=true
    engine->blas().syrk(A, C, false, true, 1.0f, 0.0f);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // C = A^T * A
    // [1 2 3]   [1 4]   [14 32]
    // [4 5 6] * [2 5] = [32 77]
    //           [3 6]
    // lower triang: (14, 32, 0, 77) -> Column-major memory: [14, 32, 0, 77]
    verify_tensor(C, {14, 32, 0, 77});
}
