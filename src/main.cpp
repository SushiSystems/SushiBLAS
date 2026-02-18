#include <iostream>
#include <SushiBLAS/SushiBLAS.h>

int main() 
{
    sr::Execution::RuntimeContext ctx;
    sb::Engine blas(ctx);

    auto A = blas.create_tensor({1024, 1024}, sr::Memory::AllocStrategy::SHARED);
    auto B = blas.create_tensor({1024, 1024}, sr::Memory::AllocStrategy::SHARED);
    auto C = blas.create_tensor({1024, 1024}, sr::Memory::AllocStrategy::SHARED);

    blas.gemm(A, B, C);

    return 0; 
}