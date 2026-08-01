/**************************************************************************/
/* example.cpp                                                            */
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
