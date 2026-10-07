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
 * @brief Fills two matrices, multiplies them with GEMM and prints one element of the result.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <iostream>
#include <SushiBLAS/SushiBLAS.h>

namespace sb = SushiBLAS;

int main()
{
    constexpr int64_t K_SIZE = 256;
    constexpr float K_A_VALUE = 1.0f;
    constexpr float K_B_VALUE = 2.0f;

    sr::Execution::RuntimeContext ctx;
    sb::Engine engine(ctx, sb::Core::Layout::ColumnMajor);

    auto A = engine.create_tensor({K_SIZE, K_SIZE});
    auto B = engine.create_tensor({K_SIZE, K_SIZE});
    auto C = engine.create_tensor({K_SIZE, K_SIZE});

    const size_t count = static_cast<size_t>(K_SIZE * K_SIZE);
    SushiBLAS::IO(engine).from_vector(A, std::vector<float>(count, K_A_VALUE));
    SushiBLAS::IO(engine).from_vector(B, std::vector<float>(count, K_B_VALUE));

    SushiBLAS::BLASOps(engine).gemm(A, B, C);

    const std::vector<float> result = SushiBLAS::IO(engine).to_vector<float>(C);

    // Every element of C is K_SIZE * K_A_VALUE * K_B_VALUE.
    std::cout << "C[0][0] = " << result[0] << " (expected " << K_SIZE * K_A_VALUE * K_B_VALUE
              << ")\n";

    return 0;
}
