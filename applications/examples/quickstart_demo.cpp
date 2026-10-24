/**************************************************************************/
/* quickstart_demo.cpp                                                    */
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
 * @file quickstart_demo.cpp
 * @brief Fills two matrices, multiplies them with GEMM and checks every element of the result.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include <SushiBLAS/SushiBLAS.hpp>

namespace sb = SushiBLAS;
namespace sr = SushiRuntime;

int main()
{
    constexpr int64_t K_SIZE = 256;
    constexpr float K_A_VALUE = 1.0f;
    constexpr float K_B_VALUE = 2.0f;

    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime, sb::Core::Layout::ColumnMajor);
    sr::Graph graph = runtime.graph();
    sb::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), engine.mode());

    auto A = engine.create_tensor({K_SIZE, K_SIZE});
    auto B = engine.create_tensor({K_SIZE, K_SIZE});
    auto C = engine.create_tensor({K_SIZE, K_SIZE});

    const size_t count = static_cast<size_t>(K_SIZE * K_SIZE);
    SushiBLAS::IO(engine).from_vector(A, std::vector<float>(count, K_A_VALUE));
    SushiBLAS::IO(engine).from_vector(B, std::vector<float>(count, K_B_VALUE));

    SushiBLAS::BLASOps(recorder).gemm(A, B, C);
    graph.run().value();

    const std::vector<float> result = SushiBLAS::IO(engine).to_vector<float>(C);

    // Every element of C is K_SIZE * K_A_VALUE * K_B_VALUE.
    const float expected = static_cast<float>(K_SIZE) * K_A_VALUE * K_B_VALUE;
    const bool correct = result == std::vector<float>(count, expected);
    std::cout << "C[0][0] = " << result[0] << " (expected " << expected << ")\n";
    std::cout << "quickstart_demo: " << (correct ? "ok" : "MISMATCH") << "\n";

    return correct ? 0 : 1;
}
