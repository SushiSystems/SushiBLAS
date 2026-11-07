/**************************************************************************/
/* engine_contract.hpp                                                    */
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
 * @file engine_contract.hpp
 * @brief Declares the checks every Engine adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor_expectations/tensor_copy.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The element count of the tensors the contract creates.
        inline constexpr std::int64_t K_ENGINE_CONTRACT_ELEMENTS = 4;

        /// @brief The value a contract reference body writes into every element.
        inline constexpr float K_ENGINE_CONTRACT_VALUE = 7.0f;

        /** @brief Returns the elements of a FLOAT32 tensor, read back through the copy helper. */
        [[nodiscard]] inline std::vector<float> floats_of(const Tensor& tensor)
        {
            std::vector<float> host(static_cast<std::size_t>(tensor.num_elements));
            copy_tensor_to_host(host.data(), tensor, host.size() * sizeof(float));
            return host;
        }

        /** @brief Writes K_ENGINE_CONTRACT_VALUE into every element of a FLOAT32 tensor. */
        inline void write_contract_value(const Tensor& tensor)
        {
            float* elements = static_cast<float*>(const_cast<Tensor&>(tensor).data());
            for (std::int64_t i = 0; i < tensor.num_elements; ++i)
                elements[i] = K_ENGINE_CONTRACT_VALUE;
        }

        /** @brief Returns whether engine.set_mode(mode) throws UnsupportedExecution. */
        [[nodiscard]] inline bool set_mode_is_refused(Engine& engine, ExecutionMode mode)
        {
            try
            {
                engine.set_mode(mode);
            }
            catch (const UnsupportedExecution&)
            {
                return true;
            }
            return false;
        }

        /** @brief Returns the message engine.set_mode(mode) refuses with, or an empty one when it accepts. */
        [[nodiscard]] inline std::string set_mode_refusal(Engine& engine, ExecutionMode mode)
        {
            try
            {
                engine.set_mode(mode);
            }
            catch (const UnsupportedExecution& refusal)
            {
                return refusal.what();
            }
            return std::string();
        }

        /** @brief Returns whether recorder.set_mode(mode) throws UnsupportedExecution. */
        [[nodiscard]] inline bool recorder_refuses(TaskRecorder& recorder, ExecutionMode mode)
        {
            try
            {
                recorder.set_mode(mode);
            }
            catch (const UnsupportedExecution&)
            {
                return true;
            }
            return false;
        }

        /** @brief Holds the checks that define an Engine; an adapter's suite calls each one. */
        struct EngineContract
        {
            /** @brief A created tensor has the shape and dtype asked for, and holds the bytes written to it. */
            template<typename Subject>
            static void expect_tensor_bytes_read_back(Subject& subject)
            {
                Tensor tensor = subject.engine().create_tensor({K_ENGINE_CONTRACT_ELEMENTS},
                                                               Core::DataType::Float32);
                const std::vector<float> written{1.0f, -2.0f, 3.5f, 0.0f};
                copy_host_to_tensor(tensor, written.data(), written.size() * sizeof(float));

                EXPECT_EQ(tensor.num_elements, K_ENGINE_CONTRACT_ELEMENTS);
                EXPECT_EQ(tensor.dtype, Core::DataType::Float32);
                EXPECT_EQ(floats_of(tensor), written);
            }

            /** @brief A recorded reference task runs on run() and writes its result into its operand. */
            template<typename Subject>
            static void expect_reference_task_records_and_runs(Subject& subject)
            {
                Tensor target = subject.engine().create_tensor({K_ENGINE_CONTRACT_ELEMENTS},
                                                               Core::DataType::Float32);
                const std::vector<float> zeros(static_cast<std::size_t>(K_ENGINE_CONTRACT_ELEMENTS),
                                               0.0f);
                copy_host_to_tensor(target, zeros.data(), zeros.size() * sizeof(float));
                TaskDescription description;
                description.name = "engine.contract";
                OperandSet operands;
                operands.writes(target);

                subject.recorder().record_pair(
                    description, operands, "Engine contract",
                    [target](const KernelPairDispatcher& pair,
                             const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return pair.run(
                            deps, [target]() { write_contract_value(target); },
                            [](sycl::queue&, const std::vector<sycl::event>&)
                            {
                                return sycl::event();
                            });
                    });

                EXPECT_EQ(subject.run(), std::string());
                EXPECT_EQ(floats_of(target),
                          std::vector<float>(static_cast<std::size_t>(K_ENGINE_CONTRACT_ELEMENTS),
                                             K_ENGINE_CONTRACT_VALUE));
            }

            /** @brief A scratch lease has the bytes asked for and returns its block to the engine pool. */
            template<typename Subject>
            static void expect_scratch_lease_and_return(Subject& subject)
            {
                ScratchPool& pool = subject.engine().scratch_pool();
                ASSERT_EQ(pool.idle_block_count(), std::size_t{0});

                {
                    const ScratchBuffer lease(pool, 300, 0);

                    EXPECT_NE(lease.data(), nullptr);
                    EXPECT_EQ(lease.bytes(), std::size_t{300});
                    EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
                }

                EXPECT_EQ(pool.idle_block_count(), std::size_t{1});
            }

            /** @brief ForceReference is accepted by every engine. */
            template<typename Subject>
            static void expect_force_reference_is_accepted(Subject& subject)
            {
                EXPECT_FALSE(set_mode_is_refused(subject.engine(), ExecutionMode::ForceReference));
                EXPECT_EQ(subject.engine().mode(), ExecutionMode::ForceReference);
            }

            /**
             * @brief set_mode keeps a mode the subject's recorder admits and refuses one it does
             *        not, leaving the mode as it was and saying the mode is not admitted.
             */
            template<typename Subject>
            static void expect_set_mode_admits_what_the_recorder_admits(Subject& subject,
                                                                        ExecutionMode mode)
            {
                const ExecutionMode before = subject.engine().mode();
                const bool recorder_refused = recorder_refuses(subject.recorder(), mode);
                const std::string refusal = set_mode_refusal(subject.engine(), mode);

                EXPECT_EQ(!refusal.empty(), recorder_refused);
                EXPECT_EQ(subject.engine().mode(), recorder_refused ? before : mode);
                EXPECT_TRUE(refusal.empty() || refusal.find("does not admit") != std::string::npos)
                    << refusal;
            }

            /** @brief The random stream reserves consecutive offsets under the seed it was set to. */
            template<typename Subject>
            static void expect_random_stream_reserves_offsets(Subject& subject)
            {
                subject.engine().rng().reseed(11);

                const RNGPosition first = subject.engine().rng().reserve(3);
                const RNGPosition second = subject.engine().rng().reserve(2);

                EXPECT_EQ(first.seed, std::uint64_t{11});
                EXPECT_EQ(first.offset, std::uint64_t{0});
                EXPECT_EQ(second.offset, std::uint64_t{3});
            }
        };
    } // namespace Test
} // namespace SushiBLAS
