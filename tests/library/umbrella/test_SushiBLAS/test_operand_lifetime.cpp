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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
#include <SushiBLAS/SushiBLAS.hpp>
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
        class Medium_OperandLifetime : public SushiBLASTest
        {
            protected:
                /** @brief Returns how many USM allocations of the test context are alive. */
                [[nodiscard]] static std::size_t live_allocations()
                {
                    return TestEnvironment::live_allocation_count();
                }
        };

        /** @brief Tests that an input destroyed after the record call lives until the task ran. */
        TEST_F(Medium_OperandLifetime, ATaskKeepsItsInputAllocatedUntilItHasRun)
        {
            auto out = engine->create_tensor({4}, Core::DataType::Float32);
            std::size_t live_with_input = 0;

            {
                const Tensor x = create_initialized_tensor<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
                live_with_input = live_allocations();
                SushiBLAS::ElementwiseOps(recorder()).add(x, x, out);
            }

            EXPECT_EQ(live_with_input, live_allocations())
                << "The recorded task is the last holder of x's storage and must keep it.";

            recording->run().value();

            expect_tensor_equal(out, std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});
            EXPECT_EQ(live_with_input - 1, live_allocations())
                << "Once the task has run and is destroyed, x's storage must be freed.";
        }

        /** @brief Tests that a GEMM epilogue's declared read outlives the tensor it came from. */
        TEST_F(Medium_OperandLifetime, AnEpilogueOperandIsKept)
        {
            auto A = create_initialized_tensor<float>({3, 2}, {1, 0, 0, 1, 1, 1});
            auto B = create_initialized_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
            auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
            std::size_t live_with_bias = 0;

            set_mode(ExecutionMode::ForceSYCL);
            {
                const Tensor bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                live_with_bias = live_allocations();

                const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 0, 1};
                Dispatch::execute_gemm(recorder(), A, B, C, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 1.0, 0.0,
                                       "test.gemm_kept_epilogue",
                                       "test.gemm_kept_epilogue"_op,
                                       Dispatch::gemm_epilogue(epilogue).reads(bias));
            }

            EXPECT_GE(live_allocations(), live_with_bias)
                << "The recorded GEMM must keep the bias its epilogue reads.";

            recording->run().value();

            // [[1,2,3],[4,5,6],[5,7,9]] plus bias[column] = [1,-8,3] down every row.
            expect_tensor_equal(C, std::vector<float>{2, -6, 6,
                                                      5, -3, 9,
                                                      6, -1, 12});
        }
    } // namespace Test
} // namespace SushiBLAS
