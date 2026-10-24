/**************************************************************************/
/* test_channel_bias_epilogue.cpp                                         */
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
 * @file test_channel_bias_epilogue.cpp
 * @brief Tests ChannelBiasEpilogue's stride pair through Dispatch::execute_gemm.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /**
             * @brief Returns the bit pattern of @p value, read without type punning.
             * @see tests/operations/blas/README.md
             */
            [[nodiscard]] uint32_t bits_of(float value)
            {
                uint32_t pattern = 0;
                std::memcpy(&pattern, &value, sizeof(pattern));
                return pattern;
            }
        } // namespace

        /**
         * @brief Holds the operands of the ChannelBiasEpilogue tests run through execute_gemm.
         * @see tests/operations/blas/README.md
         */
        class Medium_ChannelBiasEpilogueTest : public SushiBLASTest
        {
            protected:
                /** @brief A = 3x2, chosen so A*B is square and not symmetric. */
                static std::vector<float> a_values() { return {1, 0, 0, 1, 1, 1}; }

                /** @brief B = 2x3. */
                static std::vector<float> b_values() { return {1, 2, 3, 4, 5, 6}; }

                /**
                * @brief Copies a float tensor's contents out to a host vector.
                *
                * @param t The tensor to read; must hold FLOAT32 elements.
                * @return Its elements in storage order.
                */
                [[nodiscard]] static std::vector<float> download(const Tensor& t)
                {
                    std::vector<float> host(static_cast<std::size_t>(t.num_elements));
                    auto& q = TestEnvironment::queue();
                    q.memcpy(host.data(), t.data(), host.size() * sizeof(float)).wait();
                    return host;
                }
        };

        TEST_F(Medium_ChannelBiasEpilogueTest, AColumnChannelBiasMatchesAPlainRowVectorBroadcast)
        {
            // (0, 1): the channel is the column. See tests/operations/blas/README.md.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 0, 1};

            set_mode(ExecutionMode::ForceSYCL);
            Dispatch::execute_gemm(recorder(), A, B, C, Core::Transpose::NoTrans,
                                   Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.gemm_channel_bias_col",
                                   "test.gemm_channel_bias_col"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            recording->run().value();

            // [[1,2,3],[4,5,6],[5,7,9]] + bias[col] = [1,-8,3] down every row.
            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, -6, 6,
                                                      5, -3, 9,
                                                      6, -1, 12}), 1e-5);
        }

        TEST_F(Medium_ChannelBiasEpilogueTest, ARowChannelBiasBroadcastsDownTheOtherAxis)
        {
            // (1, 0): the channel is the row, as in NCHW. See tests/operations/blas/README.md.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 1, 0};

            set_mode(ExecutionMode::ForceSYCL);
            Dispatch::execute_gemm(recorder(), A, B, C, Core::Transpose::NoTrans,
                                   Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.gemm_channel_bias_row",
                                   "test.gemm_channel_bias_row"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            recording->run().value();

            // [[1,2,3],[4,5,6],[5,7,9]] + bias[row] = +1, -8, +3 across each row.
            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, 3, 4,
                                                      -4, -3, -2,
                                                      8, 10, 12}), 1e-5);
        }

        TEST_F(Medium_ChannelBiasEpilogueTest, TheSameEpilogueAppliesOnTheReferencePath)
        {
            // The reference kernel runs the epilogue too. See tests/operations/blas/README.md.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 1, 0};

            set_mode(ExecutionMode::ForceReference);
            Dispatch::execute_gemm(recorder(), A, B, C, Core::Transpose::NoTrans,
                                   Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.gemm_channel_bias_row_ref",
                                   "test.gemm_channel_bias_row_ref"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            recording->run().value();

            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, 3, 4,
                                                      -4, -3, -2,
                                                      8, 10, 12}), 1e-5);
        }

        TEST_F(Medium_ChannelBiasEpilogueTest, AZeroBiasIsTheIdentityEpilogue)
        {
            // Every element of A*B is strictly positive. See tests/operations/blas/README.md.
            auto A = create_initialized_tensor<float>({3, 2}, {1, 2, 3, 4, 5, 6});
            auto B = create_initialized_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
            auto fused = engine->create_tensor({3, 3}, Core::DataType::Float32);
            auto plain = engine->create_tensor({3, 3}, Core::DataType::Float32);
            auto bias = create_initialized_tensor<float>({3}, {0.0f, 0.0f, 0.0f});

            set_mode(ExecutionMode::ForceSYCL);
            Dispatch::execute_gemm(recorder(), A, B, fused, Core::Transpose::NoTrans,
                                   Core::Transpose::NoTrans, 1.5, 0.0,
                                   "test.gemm_zero_channel_bias",
                                   "test.gemm_zero_channel_bias"_op,
                                   Dispatch::gemm_epilogue(
                                       Kernels::ChannelBiasEpilogue{bias.data(), 0, 1})
                                       .reads(bias));
            Dispatch::execute_gemm(recorder(), A, B, plain, Core::Transpose::NoTrans,
                                   Core::Transpose::NoTrans, 1.5, 0.0,
                                   "test.gemm_zero_identity",
                                   "test.gemm_zero_identity"_op,
                                   Kernels::GemmIdentityEpilogue{});
            recording->run().value();

            const std::vector<float> fused_host = download(fused);
            const std::vector<float> plain_host = download(plain);
            ASSERT_EQ(plain_host.size(), fused_host.size());

            for (std::size_t i = 0; i < fused_host.size(); ++i)
            {
                EXPECT_EQ(bits_of(plain_host[i]), bits_of(fused_host[i]))
                    << "Element " << i << ": a zero ChannelBiasEpilogue must produce the "
                       "identity kernel's result to the bit. A difference means the fused "
                       "store changed the arithmetic, so fusion is not free when unused.";
            }
        }

        TEST_F(Medium_ChannelBiasEpilogueTest, TheEpiloguesAccessSetReachesTheTask)
        {
            GraphProbe probe;

            // The in-degree counts the bias edge alone. See tests/operations/blas/README.md.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
            auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto bias = engine->create_tensor({3}, Core::DataType::Float32);

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(seed, seed, bias);

            Dispatch::execute_gemm(probe.recorder(), A, B, C, Core::Transpose::NoTrans,
                                   Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.gemm_channel_bias_declared",
                                   "test.gemm_channel_bias_declared"_op,
                                   Dispatch::gemm_epilogue(
                                       Kernels::ChannelBiasEpilogue{bias.data(), 0, 1})
                                       .reads(bias));

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                << "The GEMM whose ChannelBiasEpilogue dereferences `bias` must depend on "
                   "the op that wrote it. An in-degree of 0 means the declaration made "
                   "through Dispatch::gemm_epilogue never reached the task's read set, and "
                   "the GEMM is free to run against a bias that has not arrived.";
        }

        TEST_F(Medium_ChannelBiasEpilogueTest, TheGemmSeesTheProducedBiasValues)
        {
            // The GEMM must see the produced bias values. See tests/operations/blas/README.md.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
            auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto bias = engine->create_tensor({3}, Core::DataType::Float32);

            // bias = seed*seed = [1, 4, 9], a value no uninitialised buffer would hold.
            SushiBLAS::ElementwiseOps(recorder()).mul(seed, seed, bias);

            set_mode(ExecutionMode::ForceSYCL);
            Dispatch::execute_gemm(recorder(), A, B, C, Core::Transpose::NoTrans,
                                   Core::Transpose::NoTrans, 1.0, 0.0,
                                   "test.gemm_channel_bias_produced",
                                   "test.gemm_channel_bias_produced"_op,
                                   Dispatch::gemm_epilogue(
                                       Kernels::ChannelBiasEpilogue{bias.data(), 0, 1})
                                       .reads(bias));
            recording->run().value();

            // [[1,2,3],[4,5,6],[5,7,9]] + [1,4,9] down every row.
            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, 6, 12,
                                                      5, 9, 15,
                                                      6, 11, 18}), 1e-5);
        }

    } // namespace Test
} // namespace SushiBLAS
