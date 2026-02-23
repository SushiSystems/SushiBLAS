/**************************************************************************/
/* main.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
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

#include <iostream>
#include <SushiBLAS/SushiBLAS.h>

int main() 
{
    sr::Execution::RuntimeContext ctx;
    sb::Engine sb(ctx, sb::Core::Layout::COLUMN_MAJOR);

    SB_LOG_INFO("Creating tensors for GEMM...");
    auto A = sb.create_tensor({1024, 1024}, sr::Memory::AllocStrategy::SHARED);
    auto B = sb.create_tensor({1024, 1024}, sr::Memory::AllocStrategy::SHARED);
    auto C = sb.create_tensor({1024, 1024}, sr::Memory::AllocStrategy::SHARED);

    SB_LOG_INFO("Executing GEMM operation...");
    sb.blas().gemm(A, B, C);

    // Executing the queued tasks
    sb.execute();

    // Synchronize to wait for GPU/CPU task completion
    ctx.wait_all();

    SB_LOG_INFO("SushiBLAS execution finished successfully.");

    return 0; 
}