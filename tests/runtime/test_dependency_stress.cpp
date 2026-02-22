#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <SushiBLAS/SushiBLAS.h>
#include "../test_common.hpp"

class DependencyStressTest : public SushiBLASTest {};

TEST_F(DependencyStressTest, ManyReadersOneWriter) 
{
    // Test Case 1: The "Vector Expansion" Bomb
    // A single tensor is written to, and then read by thousands of independent tasks.
    // This targets the std::vector<Node*> readers reallocation inside the DependencyTracker's spinlock.
    
    SB_LOG_INFO("Starting DependencyStressTest.ManyReadersOneWriter");
    
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

    // Submit a write (the writer_tensor is C, so it will be written to)
    // Actually we can just say dummy_weights * dummy_weights = writer_tensor
    engine->blas().gemm(dummy_weights, dummy_weights, writer_tensor);
    
    // Now submit 5000 reads! (writer_tensor is A, so it is read)
    for (int i = 0; i < num_readers; ++i) 
    {
        engine->blas().gemm(writer_tensor, dummy_weights, reader_outputs[i]);
    }
    
    SB_LOG_INFO("Executing DAG with massive reader fan-out...");
    auto start = std::chrono::high_resolution_clock::now();
    engine->execute().wait();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    SB_LOG_INFO("ManyReadersOneWriter took: {} ms", duration.count());
    
    EXPECT_GT(duration.count(), 0.0);
    SB_LOG_INFO("ManyReadersOneWriter finished test body.");
}

TEST_F(DependencyStressTest, ShardCollisionStress) 
{
    // Test Case 2: Shard Hash Collisions
    // DependencyTracker uses (ptr >> 6) % 1024. If we create thousands of independent tensors, 
    // many will fall into the same shard, causing false dependencies and heavy spinlock contention.
    
    SB_LOG_INFO("Starting DependencyStressTest.ShardCollisionStress");
    
    const int num_ops = 3000;
    const int N = 8;
    
    std::vector<sb::Tensor> As, Bs, Cs;
    for (int i = 0; i < num_ops; ++i) 
    {
        As.push_back(engine->create_tensor({N, N}));
        Bs.push_back(engine->create_tensor({N, N}));
        Cs.push_back(engine->create_tensor({N, N}));
        // Skipping fill_tensor for performance, we just want to stress the tracker
    }
    
    SB_LOG_INFO("Enqueueing {} completely independent GEMM operations...", num_ops);
    for (int i = 0; i < num_ops; ++i) 
    {
        engine->blas().gemm(As[i], Bs[i], Cs[i]);
    }
    
    SB_LOG_INFO("Executing DAG. Watch for false dependency serialization...");
    auto start = std::chrono::high_resolution_clock::now();
    engine->execute().wait();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    SB_LOG_INFO("ShardCollisionStress took: {} ms", duration.count());
    
    EXPECT_GT(duration.count(), 0.0);
    SB_LOG_INFO("ShardCollisionStress finished test body.");
}
