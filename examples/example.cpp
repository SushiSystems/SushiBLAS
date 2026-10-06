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
 * @file example.cpp
 * @brief Runs one 1024x1024 GEMM on shared memory with a column-major engine.
 * @author Mustafa Garip
 */

#include <iostream>
#include <SushiBLAS/SushiBLAS.h>

int main() 
{
    sr::Execution::RuntimeContext ctx;
    sb::Engine sb(ctx, sb::Core::Layout::COLUMN_MAJOR);

    SB_LOG_INFO("Creating tensors for GEMM...");
    auto A = sb.create_tensor({1024, 1024}, sr::Memory::Strategies::Shared::instance());
    auto B = sb.create_tensor({1024, 1024}, sr::Memory::Strategies::Shared::instance());
    auto C = sb.create_tensor({1024, 1024}, sr::Memory::Strategies::Shared::instance());

    SB_LOG_INFO("Executing GEMM operation...");
    sb.blas().gemm(A, B, C);

    // Executing the queued tasks
    sb.execute();

    // Synchronize to wait for GPU/CPU task completion
    ctx.wait_all();

    SB_LOG_INFO("SushiBLAS execution finished successfully.");

    return 0; 
}
