/**************************************************************************/
/* test_gemm_recording.cpp                                                */
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
 * @file test_gemm_recording.cpp
 * @brief Tests the dependencies Dispatch::record_gemm declares for the operands of an epilogue.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas/gemm_recording.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/graph_probe/host_graph_probe.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include "test_blas/gemm_seam_fixtures.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;
        using Core::DataType;
        using Core::Transpose;

        /**
         * @brief Reads the dependencies the fused seam declares, on host tensors and no runtime.
         *
         * The calls go through Dispatch::record_gemm and the in-memory port, so the task carries
         * the access set the seam built and no kernel.
         */
        class Small_GemmSeamDependencies : public HostEngineTest, protected GemmSeamData
        {
            protected:
                /**
                 * @brief Records C = A * B on the probe through the in-memory port.
                 * @param epilogue A bare functor or a Dispatch::gemm_epilogue bundle.
                 */
                template<typename Epilogue>
                void record(HostGraphProbe& probe, const Tensor& A, const Tensor& B, Tensor& C,
                            const char* name, SushiRuntime::OperationID op,
                            const Epilogue& epilogue)
                {
                    Dispatch::InMemoryGemmPort port;
                    Dispatch::record_gemm(port, probe.recorder(), A, B, C, Transpose::NoTrans,
                                          Transpose::NoTrans, 1.0, 0.0, name, op, true,
                                          Dispatch::epilogue_access(epilogue));
                }
        };

        TEST_F(Small_GemmSeamDependencies, ADeclaredEpilogueReadIsOrderedBehindItsProducer)
        {
            HostGraphProbe probe;

            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);
            auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto bias = engine->create_tensor({3}, DataType::Float32);

            // The in-degree counts the bias edge alone.
            SushiBLAS::ElementwiseOps(probe.recorder()).mul(seed, seed, bias);

            record(probe, A, B, C, "test.gemm_declared_bias", "test.gemm_declared_bias"_op,
                   Dispatch::gemm_epilogue(BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));

            ASSERT_EQ(2u, probe.recorded_count());
            const std::vector<RecordedTask> tasks = probe.recorded_tasks();
            EXPECT_EQ(1u, tasks[1].dependency_count)
                << "The GEMM whose epilogue dereferences `bias` must depend on the op "
                   "that wrote it. An in-degree of 0 means the declaration made through "
                   "Dispatch::gemm_epilogue never reached the task's read set.";
        }

        TEST_F(Small_GemmSeamDependencies, AChannelBiasEpilogueReadIsOrderedBehindItsProducer)
        {
            HostGraphProbe probe;

            // The in-degree counts the bias edge alone.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);
            auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto bias = engine->create_tensor({3}, DataType::Float32);

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(seed, seed, bias);

            record(probe, A, B, C, "test.gemm_channel_bias_declared",
                   "test.gemm_channel_bias_declared"_op,
                   Dispatch::gemm_epilogue(Kernels::ChannelBiasEpilogue{bias.data(), 0, 1})
                       .reads(bias));

            ASSERT_EQ(2u, probe.recorded_count());
            const std::vector<RecordedTask> tasks = probe.recorded_tasks();
            EXPECT_EQ(1u, tasks[1].dependency_count)
                << "The GEMM whose ChannelBiasEpilogue dereferences `bias` must depend on "
                   "the op that wrote it. An in-degree of 0 means the declaration made "
                   "through Dispatch::gemm_epilogue never reached the task's read set, and "
                   "the GEMM is free to run against a bias that has not arrived.";
        }

        TEST_F(Small_GemmSeamDependencies, AnUndeclaredEpilogueOperandGetsNoEdgeAtAll)
        {
            HostGraphProbe probe;

            // The same functor passed bare gets no edge.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);
            auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto bias = engine->create_tensor({3}, DataType::Float32);

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(seed, seed, bias);

            record(probe, A, B, C, "test.gemm_bare_bias", "test.gemm_bare_bias"_op,
                   BiasColumnEpilogue{bias.data_as<float>()});

            ASSERT_EQ(2u, probe.recorded_count());
            const std::vector<RecordedTask> tasks = probe.recorded_tasks();
            EXPECT_EQ(0u, tasks[1].dependency_count)
                << "A bare epilogue declares nothing, so the GEMM must be a root - "
                   "unordered against the op writing the bias it reads. If this is 1, "
                   "something is inferring the epilogue's operands, which this layer "
                   "cannot do correctly and must not appear to do.";
        }

        TEST_F(Small_GemmSeamDependencies, DeclaredOperandsOrderAcrossTwoViewsOfOneAllocation)
        {
            HostGraphProbe probe;

            // The declared read uses Tensor::region().
            constexpr int64_t K_COLUMNS = 3;

            auto storage = make_host_storage(static_cast<size_t>(2 * K_COLUMNS) * sizeof(float));

            Tensor seed(storage, {K_COLUMNS}, DataType::Float32, 0);
            Tensor bias(storage, {K_COLUMNS}, DataType::Float32, K_COLUMNS);

            fill_tensor(seed, {1.0f, 2.0f, 3.0f});

            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(seed, seed, bias);

            record(probe, A, B, C, "test.gemm_arena_bias", "test.gemm_arena_bias"_op,
                   Dispatch::gemm_epilogue(BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));

            ASSERT_EQ(2u, probe.recorded_count());
            const std::vector<RecordedTask> tasks = probe.recorded_tasks();
            EXPECT_EQ(1u, tasks[1].dependency_count)
                << "The declared read must be described by Tensor::region() - keyed on "
                   "the allocation base and narrowed to the view's byte interval. An "
                   "in-degree of 0 means it was described by a bare pointer, which keys "
                   "on a base nothing else in this graph shares.";
        }

        TEST_F(Small_GemmSeamDependencies, ADeclaredEpilogueWriteOrdersItsReaderBehindTheGemm)
        {
            HostGraphProbe probe;

            // Only a declared write orders the aux reader.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
            auto aux = engine->create_tensor({3, 3}, DataType::Float32);
            auto consumer = engine->create_tensor({3, 3}, DataType::Float32);

            record(probe, A, B, C, "test.gemm_spill", "test.gemm_spill"_op,
                   Dispatch::gemm_epilogue(
                       BiasSpillEpilogue{bias.data_as<float>(), aux.data_as<float>(), 3})
                       .reads(bias)
                       .writes(aux));

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(aux, aux, consumer);

            ASSERT_EQ(2u, probe.recorded_count());
            const std::vector<RecordedTask> tasks = probe.recorded_tasks();
            EXPECT_EQ(1u, tasks[1].dependency_count)
                << "The op reading the epilogue's spill buffer must depend on the GEMM "
                   "that filled it. An in-degree of 0 means the declared write never "
                   "reached the task's write set, and the spill buys nothing.";
        }

        TEST_F(Small_GemmSeamDependencies, AnUndeclaredEpilogueWriteGetsNoEdgeAtAll)
        {
            HostGraphProbe probe;

            // An undeclared aux write gets no edge.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
            auto aux = engine->create_tensor({3, 3}, DataType::Float32);
            auto consumer = engine->create_tensor({3, 3}, DataType::Float32);

            record(probe, A, B, C, "test.gemm_spill_bare", "test.gemm_spill_bare"_op,
                   BiasSpillEpilogue{bias.data_as<float>(), aux.data_as<float>(), 3});

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(aux, aux, consumer);

            ASSERT_EQ(2u, probe.recorded_count());
            const std::vector<RecordedTask> tasks = probe.recorded_tasks();
            EXPECT_EQ(0u, tasks[1].dependency_count)
                << "Nothing can see an undeclared write, so the reader must be a root.";
        }

        /** @brief Reads what the port records for a call and what it refuses, with no kernel. */
        class Small_GemmRecording : public HostEngineTest, protected GemmSeamData
        {

        };

        TEST_F(Small_GemmRecording, ACallRecordsOneTaskNamedAfterTheCall)
        {
            HostGraphProbe probe;
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, DataType::Float32);

            Dispatch::InMemoryGemmPort port;
            Dispatch::record_gemm(port, probe.recorder(), A, B, C, Transpose::NoTrans,
                                  Transpose::NoTrans, 1.0, 0.0, "test.gemm_named",
                                  "test.gemm_named"_op, true, Dispatch::GemmEpilogueAccess{});

            ASSERT_EQ(1u, probe.recorded_count());
            EXPECT_STREQ("test.gemm_named", probe.recorded_tasks()[0].name.c_str());
        }

        TEST_F(Small_GemmRecording, ARefusedCallRecordsNoTask)
        {
            HostGraphProbe probe;
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = engine->create_tensor({2, 3}, DataType::Float64);
            auto C = engine->create_tensor({3, 3}, DataType::Float32);

            Dispatch::InMemoryGemmPort port;
            EXPECT_THROW(Dispatch::record_gemm(port, probe.recorder(), A, B, C, Transpose::NoTrans,
                                               Transpose::NoTrans, 1.0, 0.0, "test.gemm_refused",
                                               "test.gemm_refused"_op, true,
                                               Dispatch::GemmEpilogueAccess{}),
                         std::invalid_argument);
            EXPECT_EQ(0u, probe.recorded_count());
        }
    } // namespace Test
} // namespace SushiBLAS
