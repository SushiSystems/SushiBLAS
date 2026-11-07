/**************************************************************************/
/* engine.hpp                                                             */
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
 * @file engine.hpp
 * @brief Declares the Engine, which owns what a process shares between its graphs.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/mode_admission.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/scratch_block_source.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    /**
     * @class Engine
     * @brief Owns a scratch pool over an owned block source, a tensor
     *        factory, a random stream and an owned mode admission; it records and runs no graph.
     *        Its two constructors are its composition roots, for the runtime and the host.
     * @see docs/design/LAYERED_EXECUTION/README.md
     */
    class Engine final
    {
        public:
            /**
             * @brief Builds an engine on a runtime, which wires its tensor factory and scratch source.
             * @param runtime Must outlive the engine; the engine keeps no reference to it.
             * @param layout The memory layout of tensors this engine creates.
             */
            explicit Engine(SushiRuntime::Runtime& runtime,
                            Core::Layout layout = Core::Layout::RowMajor);

            /** @brief Selects the constructor of an engine that needs no runtime. */
            struct Host
            {

            };

            /**
             * @brief Builds a host engine: a host tensor factory, a host scratch source, a host
             *        mode admission and a random stream, with mode ForceReference. A recorder on
             *        it records through a HostTaskSink its owner builds.
             * @param layout The memory layout of tensors this engine creates.
             * @see modules/engine/engine/README.md
             */
            explicit Engine(Host, Core::Layout layout = Core::Layout::RowMajor);

            Engine(const Engine&) = delete;
            Engine& operator=(const Engine&) = delete;
            Engine(Engine&&) = delete;
            Engine& operator=(Engine&&) = delete;
            ~Engine();

            /** @brief Returns the factory create_tensor forwards to. */
            [[nodiscard]] TensorFactory& tensor_factory() noexcept { return tensor_factory_; }

            /**
             * @brief Returns the random stream a RandomOps view built with it draws from.
             * @see modules/engine/engine/README.md
             */
            [[nodiscard]] RNGStream& rng() noexcept { return rng_; }

            /**
             * @brief Returns the pool tasks recorded over it lease device scratch from.
             * @see modules/engine/engine/README.md
             */
            [[nodiscard]] ScratchPool& scratch_pool() noexcept { return scratch_pool_; }

            /**
             * @brief Sets the execution mode a recorder built from now on starts with.
             * @pre No thread is building a recorder from this engine.
             * @throws UnsupportedExecution when the engine's mode admission refuses the mode;
             *         the mode stays.
             */
            void set_mode(ExecutionMode mode)
            {
                if (!admission_->admits(mode))
                {
                    throw UnsupportedExecution(
                        "This Engine does not admit the requested execution mode; its mode stays.");
                }
                mode_ = mode;
            }

            /** @brief Returns the execution mode a recorder built from this engine starts with. */
            [[nodiscard]] ExecutionMode mode() const noexcept { return mode_; }

            /** @brief Returns the memory layout of tensors this engine creates. */
            [[nodiscard]] Core::Layout layout() const noexcept
            {
                return tensor_factory_.layout();
            }

            /**
             * @brief Creates a tensor from a brace list of dimensions.
             * @param device_index Device queue that owns the allocation.
             * @return A tensor without storage when a dimension is 0.
             * @throws std::runtime_error as TensorFactory::create states.
             */
            [[nodiscard]] Tensor create_tensor(
                std::initializer_list<int64_t> dims,
                Core::DataType dtype = Core::DataType::Float32,
                SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                std::size_t device_index = 0)
            {
                return tensor_factory_.create(
                    SushiRuntime::span<const int64_t>(dims.begin(), dims.end()), dtype, residency,
                    device_index);
            }

            /**
             * @brief Creates a tensor whose dimensions are known only at run time.
             * @param device_index Device queue that owns the allocation.
             * @return A tensor without storage when a dimension is 0.
             * @throws std::runtime_error as TensorFactory::create states.
             */
            [[nodiscard]] Tensor create_tensor(
                SushiRuntime::span<const int64_t> dims,
                Core::DataType dtype = Core::DataType::Float32,
                SushiRuntime::Residency residency = SushiRuntime::Residency::Shared,
                std::size_t device_index = 0)
            {
                return tensor_factory_.create(dims, dtype, residency, device_index);
            }

        private:
            /// @brief The source of every scratch block the pool lends; never null.
            std::unique_ptr<ScratchBlockSource> block_source_;

            /// @brief Answers which modes set_mode accepts; never null.
            std::unique_ptr<ModeAdmission> admission_;

            /// @brief The pool recorded tasks lease scratch from; see modules/README.md.
            ScratchPool scratch_pool_;

            /// @brief The execution mode a recorder built from this engine starts with.
            ExecutionMode mode_ = ExecutionMode::Auto;

            /// @brief The factory behind create_tensor.
            TensorFactory tensor_factory_;

            /// @brief The random stream of this engine.
            RNGStream rng_;
    };
} // namespace SushiBLAS
