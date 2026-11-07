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
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    class HostScratchSource;
    class HostTaskSink;

    /**
     * @class Engine
     * @brief Owns a runtime binding, a scratch pool, a tensor factory, a random stream and an
     *        execution mode; it records and runs no graph. A host engine has no runtime and
     *        owns the HostTaskSink its recorders record through.
     * @see docs/design/LAYERED_EXECUTION/README.md
     * @see modules/README.md
     */
    class Engine final
    {
        public:
            /**
             * @brief Builds an engine on a runtime.
             * @param runtime Must outlive the engine.
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
             *        task sink and a random stream, with mode ForceReference.
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

            /** @brief Returns whether the engine is a host engine and has no runtime. */
            [[nodiscard]] bool is_host() const noexcept { return runtime_ == nullptr; }

            /**
             * @brief Returns the runtime the engine was built on.
             * @throws UnsupportedExecution on a host engine, which has none.
             */
            [[nodiscard]] SushiRuntime::Runtime& runtime()
            {
                if (runtime_ == nullptr)
                    throw UnsupportedExecution("A host Engine has no runtime");
                return *runtime_;
            }

            /**
             * @brief Returns the sink a TaskRecorder of a host engine records through.
             * @throws UnsupportedExecution on an engine built on a runtime, whose recorders
             *         record on a graph.
             */
            [[nodiscard]] HostTaskSink& host_sink()
            {
                if (host_sink_ == nullptr)
                    throw UnsupportedExecution("An Engine built on a runtime has no host task sink");
                return *host_sink_;
            }

            /**
             * @brief Sets the execution mode a recorder built from now on starts with.
             * @pre No thread is building a recorder from this engine.
             * @throws UnsupportedExecution on a host engine for any mode but ForceReference; the
             *         mode stays.
             */
            void set_mode(ExecutionMode mode)
            {
                if (is_host() && mode != ExecutionMode::ForceReference)
                {
                    throw UnsupportedExecution(
                        "A host Engine runs ForceReference only: it has no SYCL queue.");
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
            /// @brief The runtime that allocates and executes; null on a host engine.
            SushiRuntime::Runtime* runtime_;

            /// @brief The source of the scratch blocks of a host engine; empty otherwise.
            std::unique_ptr<HostScratchSource> host_source_;

            /// @brief The pool recorded tasks lease scratch from; see modules/README.md.
            ScratchPool scratch_pool_;

            /// @brief The execution mode a recorder built from this engine starts with.
            ExecutionMode mode_ = ExecutionMode::Auto;

            /// @brief The factory behind create_tensor.
            TensorFactory tensor_factory_;

            /// @brief The random stream of this engine.
            RNGStream rng_;

            /// @brief The sink a host engine's recorders record through; empty otherwise.
            std::unique_ptr<HostTaskSink> host_sink_;
    };
} // namespace SushiBLAS
