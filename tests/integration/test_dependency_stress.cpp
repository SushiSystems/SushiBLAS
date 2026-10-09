/**************************************************************************/
/* test_dependency_stress.cpp                                             */
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
 * @file test_dependency_stress.cpp
 * @brief Tests the dependency tracker under reader fan-out and shard collisions.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <SushiBLAS/SushiBLAS.h>
#include "../common/test_utils.hpp"

namespace sb = SushiBLAS;
using namespace SushiBLAS::Test;

class Integration_DependencyStressTest : public SushiBLASTest {};

/**
 * @brief Tests that 5000 readers of one written tensor all see the written value.
 * @see tests/integration/README.md
 */
TEST_F(Integration_DependencyStressTest, ManyReadersOneWriter) 
{
    
    SB_LOG_INFO("Starting Integration_DependencyStressTest.ManyReadersOneWriter");
    
    const int num_readers = 5000;
    const int N = 32;
    
    auto writer_tensor = engine->create_tensor({N, N});
    fill_tensor(writer_tensor, std::vector<float>(N * N, 1.0f));
    
    std::vector<sb::Tensor> reader_outputs;
    for (int i = 0; i < num_readers; ++i) 
    {
        auto t = engine->create_tensor({N, N});
        fill_tensor(t, std::vector<float>(N * N, 0.0f));
        reader_outputs.push_back(t);
    }
    
    SB_LOG_INFO("Enqueueing a single writer and {} readers...", num_readers);
    
    // Create an independent tensor just for the GEMM to multiply with writer_tensor
    auto dummy_weights = engine->create_tensor({N, N});
    fill_tensor(dummy_weights, std::vector<float>(N * N, 2.0f));

    // writer_tensor is the GEMM's output, so this call writes it.
    SushiBLAS::BLASOps(*engine).gemm(dummy_weights, dummy_weights, writer_tensor);
    
    // Now submit 5000 reads! (writer_tensor is A, so it is read)
    for (int i = 0; i < num_readers; ++i) 
    {
        SushiBLAS::BLASOps(*engine).gemm(writer_tensor, dummy_weights, reader_outputs[i]);
    }
    
    SB_LOG_INFO("Executing DAG with massive reader fan-out...");
    auto start = std::chrono::high_resolution_clock::now();
    engine->execute();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    SB_LOG_INFO("ManyReadersOneWriter took: {} ms", duration.count());
    
    // writer = 32 * 2 * 2 = 128 everywhere; every reader = 32 * 128 * 2 = 8192.
    const std::vector<float> K_EXPECTED_READER(N * N, 8192.0f);
    for (int i = 0; i < num_readers; ++i)
        verify_tensor(reader_outputs[i], K_EXPECTED_READER, 0.0);
    SB_LOG_INFO("ManyReadersOneWriter finished test body.");
}

/**
 * @brief Tests that 3000 independent GEMMs on distinct tensors each produce their own result.
 * @see tests/integration/README.md
 */
TEST_F(Integration_DependencyStressTest, ShardCollisionStress) 
{
    
    SB_LOG_INFO("Starting Integration_DependencyStressTest.ShardCollisionStress");
    
    const int num_ops = 3000;
    const int N = 8;
    const float K_B_VALUE = 0.5f;
    const auto a_value_of = [](int i) { return static_cast<float>(i % 251 + 1); };
    
    std::vector<sb::Tensor> As, Bs, Cs;
    for (int i = 0; i < num_ops; ++i) 
    {
        As.push_back(engine->create_tensor({N, N}));
        Bs.push_back(engine->create_tensor({N, N}));
        Cs.push_back(engine->create_tensor({N, N}));
        fill_tensor(As[i], std::vector<float>(N * N, a_value_of(i)));
        fill_tensor(Bs[i], std::vector<float>(N * N, K_B_VALUE));
    }
    
    SB_LOG_INFO("Enqueueing {} completely independent GEMM operations...", num_ops);
    for (int i = 0; i < num_ops; ++i) 
    {
        SushiBLAS::BLASOps(*engine).gemm(As[i], Bs[i], Cs[i]);
    }
    
    SB_LOG_INFO("Executing DAG. Watch for false dependency serialization...");
    auto start = std::chrono::high_resolution_clock::now();
    engine->execute();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    SB_LOG_INFO("ShardCollisionStress took: {} ms", duration.count());
    
    // Every element of C_i is N * a_i * K_B_VALUE = 4 * a_i, exact in float.
    for (int i = 0; i < num_ops; ++i)
        verify_tensor(Cs[i], std::vector<float>(N * N, N * a_value_of(i) * K_B_VALUE), 0.0);
    SB_LOG_INFO("ShardCollisionStress finished test body.");
}
