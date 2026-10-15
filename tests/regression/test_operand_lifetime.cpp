/**************************************************************************/
/* test_operand_lifetime.cpp                                              */
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
 * @file test_operand_lifetime.cpp
 * @brief Tests that a recorded task keeps its operands and its scratch allocated until it has run.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        /** @brief Counts the context's live USM allocations around record and execute calls. */
        class Regression_OperandLifetime : public SushiBLASTest
        {
            protected:
                /** @brief Returns how many USM allocations of the test context are alive. */
                [[nodiscard]] static std::size_t live_allocations()
                {
                    return TestEnvironment::live_allocation_count();
                }
        };

        /** @brief Tests that an input destroyed after the record call lives until the task ran. */
        TEST_F(Regression_OperandLifetime, ATaskKeepsItsInputAllocatedUntilItHasRun)
        {
            auto out = engine->create_tensor({4}, Core::DataType::Float32);
            std::size_t live_with_input = 0;

            {
                const Tensor x = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
                live_with_input = live_allocations();
                SushiBLAS::ElementwiseOps(*engine).add(x, x, out);
            }

            EXPECT_EQ(live_with_input, live_allocations())
                << "The recorded task is the last holder of x's storage and must keep it.";

            engine->execute().value();

            EXPECT_TENSOR_NEAR(out, (std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f}), 1e-5);
            EXPECT_EQ(live_with_input - 1, live_allocations())
                << "Once the task has run and is destroyed, x's storage must be freed.";
        }

        /** @brief Tests that a GEMM epilogue's declared read outlives the tensor it came from. */
        TEST_F(Regression_OperandLifetime, AnEpilogueOperandIsKept)
        {
            auto A = create_initialized_tensor<float>({3, 2}, {1, 0, 0, 1, 1, 1});
            auto B = create_initialized_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
            auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
            std::size_t live_with_bias = 0;

            engine->set_mode(ExecutionMode::ForceSYCL);
            {
                const Tensor bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                live_with_bias = live_allocations();

                const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 0, 1};
                Dispatch::execute_gemm(engine->recorder(), A, B, C, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 1.0, 0.0,
                                       "test.gemm_kept_epilogue",
                                       "test.gemm_kept_epilogue"_op,
                                       Dispatch::gemm_epilogue(epilogue).reads(bias));
            }

            EXPECT_GE(live_allocations(), live_with_bias)
                << "The recorded GEMM must keep the bias its epilogue reads.";

            engine->execute().value();

            // [[1,2,3],[4,5,6],[5,7,9]] plus bias[column] = [1,-8,3] down every row.
            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, -6, 6,
                                                      5, -3, 9,
                                                      6, -1, 12}), 1e-5);
        }

        /** @brief Tests that a reduction's scratch returns to the pool and is lent again. */
        TEST_F(Regression_OperandLifetime, AScratchReturnsToThePoolWithItsTask)
        {
            constexpr int64_t K_ELEMENTS =
                static_cast<int64_t>(Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS) + 1;

            auto values = create_initialized_tensor<float>(
                {K_ELEMENTS}, std::vector<float>(static_cast<std::size_t>(K_ELEMENTS), 1.0f));
            auto total = engine->create_tensor({1}, Core::DataType::Float32);

            ScratchPool& pool = engine->scratch_pool();
            const std::size_t outstanding_before = live_allocations() - pool.idle_block_count();

            SushiBLAS::ReductionOps(*engine).sum(values, total);
            engine->execute().value();

            EXPECT_EQ(outstanding_before, live_allocations() - pool.idle_block_count())
                << "Every allocation the sum made must be idle in the pool or freed.";
            EXPECT_TENSOR_NEAR(total, (std::vector<float>{static_cast<float>(K_ELEMENTS)}), 1e-3);

            const std::size_t live_after_first = live_allocations();
            const std::size_t idle_after_first = pool.idle_block_count();

            SushiBLAS::ReductionOps(*engine).sum(values, total);
            engine->execute().value();

            EXPECT_EQ(live_after_first, live_allocations());
            EXPECT_EQ(idle_after_first, pool.idle_block_count());
        }
    } // namespace Test
} // namespace SushiBLAS
