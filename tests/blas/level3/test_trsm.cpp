/**************************************************************************/
/* test_trsm.cpp                                                          */
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


class TRSMTest : public SushiBLASTest {};

TEST_F(TRSMTest, SimpleTRSM) 
{
    const int M = 2;
    const int N = 2;
    // A must be M x M since we will do left-side multiplication
    auto A = engine->create_tensor({M, M});
    auto B = engine->create_tensor({M, N});

    // A: Lower triangular
    // [2 0]
    // [1 2]
    fill_tensor(A, {2, 0, 1, 2});
    
    // B: 
    // [4 6]
    // [4 7]
    fill_tensor(B, {4, 6, 4, 7});

    SB_LOG_INFO("Submitting standard TRSM.");
    // Left=true, Upper=false, TransA=false, UnitDiag=false, alpha=1.0
    engine->blas().trsm(A, B, true, false, false, false, 1.0f);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // We solve A * X = B
    // [2 0] * [x11 x12] = [4 6]
    // [1 2]   [x21 x22]   [4 7]
    // 
    // 2*x11 = 4 => x11 = 2
    // 2*x12 = 6 => x12 = 3
    // 1*x11 + 2*x21 = 4 => 2 + 2*x21 = 4 => x21 = 1
    // 1*x12 + 2*x22 = 7 => 3 + 2*x22 = 7 => x22 = 2
    // 
    // X (which overwrites B) should be:
    // [2 3]
    // [1 2]
    verify_tensor(B, {2, 3, 1, 2});
}

TEST_F(TRSMTest, SimpleTRSMColumnMajor) 
{
    reinit_engine(sb::Core::Layout::COLUMN_MAJOR);
    
    const int M = 2, N = 2;
    auto A = engine->create_tensor({M, M});
    auto B = engine->create_tensor({M, N});

    // A: Upper triangular (Column-Major memory layout)
    // [2 1]
    // [0 2]
    // Col-major: [2 0 1 2]
    fill_tensor(A, {2, 0, 1, 2});
    
    // B: 
    // [4 6]
    // [4 7]
    // Col-major: [4 4 6 7]
    fill_tensor(B, {4, 4, 6, 7});

    SB_LOG_INFO("Submitting Column-Major TRSM.");
    // Left=true, Upper=true, TransA=false, UnitDiag=false, alpha=1.0
    engine->blas().trsm(A, B, true, true, false, false, 1.0f);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Solve: A * X = B
    // [2 1] * [x11 x12] = [4 6]
    // [0 2]   [x21 x22]   [4 7]
    //
    // 2*x21 = 4 => x21 = 2
    // 2*x22 = 7 => x22 = 3.5
    // 2*x11 + 1*x21 = 4 => 2*x11 + 2 = 4 => x11 = 1
    // 2*x12 + 1*x22 = 6 => 2*x12 + 3.5 = 6 => 2*x12 = 2.5 => x12 = 1.25
    // 
    // X = 
    // [1.00 1.25]
    // [2.00 3.50]
    // Col-major memory: [1.0, 2.0, 1.25, 3.5]
    verify_tensor(B, {1.0f, 2.0f, 1.25f, 3.5f});
}
