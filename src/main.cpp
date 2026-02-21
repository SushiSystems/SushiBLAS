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