#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <SushiBLAS/SushiBLAS.h>
#include "../test_common.hpp"
#include <iostream>

class FrameworkSimTest : public SushiBLASTest {};

TEST_F(FrameworkSimTest, DeepForwardPass) 
{
    // Simulates a deep neural network forward pass:
    // H1 = X * W1
    // H2 = H1 * W2
    // ...
    // H_n = H_{n-1} * W_n
    
    const int batch_size = 512;
    const int features = 1024;
    const int num_layers = 10;
    
    auto X = engine->create_tensor({batch_size, features});
    fill_tensor(X, std::vector<float>(batch_size * features, 1.0f));
    SB_LOG_INFO("Sim: X tensor created with shape {}x{}", batch_size, features);
    
    std::vector<sb::Tensor> weights;
    std::vector<sb::Tensor> activations;
    
    activations.push_back(X);
    
    for (int i = 0; i < num_layers; ++i) 
    {
        auto W = engine->create_tensor({features, features});
        
        // Just fill with small values to prevent explosion
        fill_tensor(W, std::vector<float>(features * features, 0.01f));
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
        engine->blas().gemm(activations[i], weights[i], activations[i+1]);
    }
    
    // Execute all deep operations asynchronously
    SB_LOG_INFO("Sim: Executing Deep Forward Pass Task DAG and waiting...");
    engine->execute().wait();
    SB_LOG_INFO("Sim: Deep Forward Pass execution complete");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    std::cout << "[Framework Sim] Deep Forward Pass (10 Layers " << batch_size << "x" << features << ") Took: " << duration.count() << " ms\n";
    
    // Verify sizes
    EXPECT_EQ(activations.back().shape[0], batch_size);
    EXPECT_EQ(activations.back().shape[1], features);
}

TEST_F(FrameworkSimTest, ParallelBatchedInference) 
{
    // Simulates serving multiple independent models at the same time
    // Tests the ability of the scheduler to distribute completely independent tasks 
    // across all 12 workers optimally.
    
    const int num_models = 24; // More models than threads
    const int m = 256;
    const int n = 512;
    const int k = 256;
    
    std::vector<sb::Tensor> inputs;
    std::vector<sb::Tensor> weights;
    std::vector<sb::Tensor> outputs;
    
    for (int i = 0; i < num_models; ++i) 
    {
        auto X = engine->create_tensor({m, k});
        fill_tensor(X, std::vector<float>(m * k, 1.0f));
        inputs.push_back(X);
        
        auto W = engine->create_tensor({k, n});
        fill_tensor(W, std::vector<float>(k * n, 2.0f));
        weights.push_back(W);
        
        auto Y = engine->create_tensor({m, n});
        outputs.push_back(Y);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Submit all models to the DAG
    for (int i = 0; i < num_models; ++i) 
    {
        engine->blas().gemm(inputs[i], weights[i], outputs[i]);
    }
    
    // The scheduler should dispatch these horizontally to all worker threads
    SB_LOG_INFO("Sim: Executing Parallel Inference DAG across workers...");
    engine->execute().wait();
    SB_LOG_INFO("Sim: Parallel Inference execution complete");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    std::cout << "[Framework Sim] Parallel Inference (" << num_models << " models) Took: " << duration.count() << " ms\n";
    
    for (int i = 0; i < num_models; ++i) 
    {
        EXPECT_EQ(outputs[i].shape[0], m);
        EXPECT_EQ(outputs[i].shape[1], n);
    }
}
