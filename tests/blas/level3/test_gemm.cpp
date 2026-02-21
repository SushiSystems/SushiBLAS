/**************************************************************************/
/* test_gemm.cpp                                                          */
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


class GEMMTest : public SushiBLASTest {};

TEST_F(GEMMTest, SimpleGEMM3x3) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto B = engine->create_tensor({N, N});
    auto C = engine->create_tensor({N, N});

    // Fill A: [1, 2, ..., 9]
    fill_tensor(A, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    
    // Fill B: Identity * 2.0
    fill_tensor(B, {2, 0, 0, 0, 2, 0, 0, 0, 2});
    
    // Reset C
    fill_tensor(C, std::vector<float>(N * N, 0.0f));

    engine->blas().gemm(A, B, C);
    engine->execute().wait();

    // Expected: A * 2
    verify_tensor(C, {2, 4, 6, 8, 10, 12, 14, 16, 18});
}

TEST_F(GEMMTest, SimpleGEMMColumnMajor) 
{
    reinit_engine(sb::Core::Layout::COLUMN_MAJOR);
    
    const int M = 2, N = 2, K = 2;
    auto A = engine->create_tensor({M, K});
    auto B = engine->create_tensor({K, N});
    auto C = engine->create_tensor({M, N});

    // A: [[1, 2], [3, 4]] -> Column-major: [1, 3, 2, 4]
    fill_tensor(A, {1, 3, 2, 4});
    
    // B: [[5, 6], [7, 8]] -> Column-major: [5, 7, 6, 8]
    fill_tensor(B, {5, 7, 6, 8});

    engine->blas().gemm(A, B, C);
    engine->execute().wait();

    // Expected Memory: [19, 43, 22, 50]
    verify_tensor(C, {19, 43, 22, 50});
}
