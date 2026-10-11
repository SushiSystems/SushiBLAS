/**************************************************************************/
/* test_framework_sim.cpp                                                 */
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
 * @file test_framework_sim.cpp
 * @brief Tests a ten-layer forward pass and 24 independent models through the engine.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <SushiBLAS/SushiBLAS.h>
#include "../common/test_utils.hpp"
#include <iostream>

namespace sb = SushiBLAS;
using namespace SushiBLAS::Test;

class Integration_FrameworkSimTest : public SushiBLASTest {};

/**
 * @brief Tests a chain of ten GEMMs, each followed by ReLU except the last, against a host-computed result.
 * @see tests/integration/README.md
 */
TEST_F(Integration_FrameworkSimTest, DeepForwardPass) 
{
    
    const int batch_size = 512;
    const int features = 1024;
    const int num_layers = 10;
    
    auto X = engine->create_tensor({batch_size, features});
    fill_tensor(X, std::vector<float>(batch_size * features, 1.0f));
    SB_LOG_INFO("Sim: X tensor created with shape {}x{}", batch_size, features);
    
    // Even columns of W hold +K_WEIGHT and odd columns -K_WEIGHT, so ReLU zeroes the odd ones.
    const float K_WEIGHT = 0.01f;
    std::vector<float> alternating_weights(features * features);
    for (int i = 0; i < features * features; ++i)
        alternating_weights[i] = ((i % features) % 2 == 0) ? K_WEIGHT : -K_WEIGHT;

    std::vector<sb::Tensor> weights;
    std::vector<sb::Tensor> activations;
    
    activations.push_back(X);
    
    for (int i = 0; i < num_layers; ++i) 
    {
        auto W = engine->create_tensor({features, features});
        
        fill_tensor(W, alternating_weights);
        weights.push_back(W);
        
        auto H = engine->create_tensor({batch_size, features});
        activations.push_back(H);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Build the DAG 
    SB_LOG_INFO("Sim: Building DAG for {} layers", num_layers);
    for (int i = 0; i < num_layers; ++i) 
    {
        // H_{i+1} = H_{i} * W_i
        SB_LOG_DEBUG("Sim: Submitting GEMM for Layer {}", i);
        SushiBLAS::BLASOps(*engine).gemm(activations[i], weights[i], activations[i+1]);
        
        // Add activation after GEMM
        if (i < num_layers - 1) 
        {
            SushiBLAS::NonLinearOps(*engine).relu(activations[i+1]);
        }
    }
    
    // Execute all deep operations asynchronously
    SB_LOG_INFO("Sim: Executing Deep Forward Pass Task DAG and waiting...");
    engine->execute().value();
    SB_LOG_INFO("Sim: Deep Forward Pass execution complete");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    std::cout << "[Framework Sim] Deep Forward Pass (10 Layers " << batch_size << "x" << features << ") Took: " << duration.count() << " ms\n";
    
    // Layer 0 sums all features; later layers see only the even half that survived ReLU.
    double magnitude = static_cast<double>(features) * static_cast<double>(K_WEIGHT);
    for (int i = 1; i < num_layers; ++i)
        magnitude *= static_cast<double>(features / 2) * static_cast<double>(K_WEIGHT);

    // The last layer has no ReLU, so its odd columns stay negative.
    std::vector<float> expected(batch_size * features);
    for (int i = 0; i < batch_size * features; ++i)
        expected[i] = static_cast<float>(((i % features) % 2 == 0) ? magnitude : -magnitude);

    verify_tensor(activations.back(), expected, magnitude * 1e-3);
}

/**
 * @brief Tests 24 independent GEMM and ReLU pairs submitted to one graph against their exact results.
 * @see tests/integration/README.md
 */
TEST_F(Integration_FrameworkSimTest, ParallelBatchedInference) 
{
    
    const int num_models = 24; // More models than threads
    const int m = 256;
    const int n = 512;
    const int k = 256;

    // Model i has X = i + 1 and W = +2 for even i, -2 for odd i, so ReLU zeroes the odd models.
    const auto input_of = [](int i) { return static_cast<float>(i + 1); };
    const auto weight_of = [](int i) { return (i % 2 == 0) ? 2.0f : -2.0f; };
    
    std::vector<sb::Tensor> inputs;
    std::vector<sb::Tensor> weights;
    std::vector<sb::Tensor> outputs;
    
    for (int i = 0; i < num_models; ++i) 
    {
        auto X = engine->create_tensor({m, k});
        fill_tensor(X, std::vector<float>(m * k, input_of(i)));
        inputs.push_back(X);
        
        auto W = engine->create_tensor({k, n});
        fill_tensor(W, std::vector<float>(k * n, weight_of(i)));
        weights.push_back(W);
        
        auto Y = engine->create_tensor({m, n});
        outputs.push_back(Y);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Submit all models to the DAG
    for (int i = 0; i < num_models; ++i) 
    {
        SushiBLAS::BLASOps(*engine).gemm(inputs[i], weights[i], outputs[i]);
        SushiBLAS::NonLinearOps(*engine).relu(outputs[i]);
    }
    
    // The scheduler should dispatch these horizontally to all worker threads
    SB_LOG_INFO("Sim: Executing Parallel Inference DAG across workers...");
    engine->execute().value();
    SB_LOG_INFO("Sim: Parallel Inference execution complete");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    std::cout << "[Framework Sim] Parallel Inference (" << num_models << " models) Took: " << duration.count() << " ms\n";
    
    // Every element of Y_i is relu(k * input_of(i) * weight_of(i)), exact in float.
    for (int i = 0; i < num_models; ++i)
    {
        const float pre_activation = static_cast<float>(k) * input_of(i) * weight_of(i);
        const float expected = pre_activation > 0.0f ? pre_activation : 0.0f;
        verify_tensor(outputs[i], std::vector<float>(m * n, expected), 0.0);
    }
}
