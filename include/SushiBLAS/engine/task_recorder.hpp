/**************************************************************************/
/* task_recorder.hpp                                                      */
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
 * @file task_recorder.hpp
 * @brief Declares TaskRecorder, the one caller of TaskGraph::add_task in SushiBLAS.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/graph/task_graph.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        namespace Detail
        {
            /** @brief Returns an identity no other TaskRecorder of this process holds; never 0. */
            [[nodiscard]] inline std::uint64_t next_recorder_identity() noexcept
            {
                static std::atomic<std::uint64_t> next{1};
                return next.fetch_add(1, std::memory_order_relaxed);
            }

            /**
             * @brief Runs a task body and keeps its operands' storage allocated while it exists.
             * @tparam Body Callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`.
             */
            template<typename Body>
            class RecordedTask
            {
                public:
                    /** @brief Holds the body and one reference per operand storage. */
                    RecordedTask(Body body, StorageKeepAlive retained)
                        : body_(std::move(body)), retained_(std::move(retained))
                    {

                    }

                    /** @brief Runs the body. */
                    sycl::event operator()(sycl::queue& q, const std::vector<sycl::event>& deps) const
                    {
                        return body_(q, deps);
                    }

                private:
                    /// @brief The body the operation recorded.
                    Body body_;

                    /// @brief The storages that stay allocated until the task is destroyed.
                    StorageKeepAlive retained_;
            };

            /**
             * @brief Runs a kernel-pair body with its dispatcher and keeps its operands' storage.
             * @tparam Body Callable `(const KernelPairDispatcher&, sycl::queue&,
             *         const std::vector<sycl::event>&) -> sycl::event`.
             */
            template<typename Body>
            class RecordedPairTask
            {
                public:
                    /** @brief Holds the body, its mode router and one reference per operand storage. */
                    RecordedPairTask(Body body, KernelPairDispatcher dispatcher,
                                     StorageKeepAlive retained)
                        : body_(std::move(body)), dispatcher_(dispatcher),
                          retained_(std::move(retained))
                    {

                    }

                    /** @brief Runs the body with the dispatcher fixed at record time. */
                    sycl::event operator()(sycl::queue& q, const std::vector<sycl::event>& deps) const
                    {
                        return body_(dispatcher_, q, deps);
                    }

                private:
                    /// @brief The body the operation recorded.
                    Body body_;

                    /// @brief The mode router handed to the body.
                    KernelPairDispatcher dispatcher_;

                    /// @brief The storages that stay allocated until the task is destroyed.
                    StorageKeepAlive retained_;
            };
        } // namespace Detail
    } // namespace Internal

    /**
     * @brief Records tasks on one TaskGraph under one execution mode.
     * @see include/SushiBLAS/engine/README.md
     */
    class TaskRecorder final
    {
        public:
            /**
             * @brief Binds the recorder to the graph it records on and the pool its tasks lease from.
             * @param graph Receives every task; it must outlive the recorder.
             * @param scratch_pool Lends operation scratch; it must outlive the recorder.
             */
            TaskRecorder(SushiRuntime::Graph::TaskGraph& graph, ScratchPool& scratch_pool,
                         ExecutionMode mode = ExecutionMode::AUTO) noexcept
                : graph_(&graph), scratch_pool_(&scratch_pool), mode_(mode),
                  identity_(Internal::Detail::next_recorder_identity())
            {

            }

            TaskRecorder(const TaskRecorder&) = delete;
            TaskRecorder& operator=(const TaskRecorder&) = delete;
            TaskRecorder(TaskRecorder&&) = delete;
            TaskRecorder& operator=(TaskRecorder&&) = delete;
            ~TaskRecorder() = default;

            /** @brief Returns the mode every later record call is recorded under. */
            [[nodiscard]] ExecutionMode get_mode() const noexcept { return mode_; }

            /** @brief Sets the mode every later record call is recorded under. */
            void set_mode(ExecutionMode mode) noexcept { mode_ = mode; }

            /** @brief Returns the pool recorded tasks lease their device scratch from. */
            [[nodiscard]] ScratchPool& get_scratch_pool() const noexcept
            {
                return *scratch_pool_;
            }

            /**
             * @brief Throws as record() would, for a caller that must refuse before a side effect.
             * @throws std::invalid_argument under FORCE_REFERENCE, naming family and name.
             */
            void require_sycl_path(const char* family, const char* name) const
            {
                if (mode_ != ExecutionMode::FORCE_REFERENCE)
                    return;

                const std::string message = std::string(family) + " '" + name +
                                            "' has no reference kernel, so it cannot be recorded "
                                            "under ExecutionMode::FORCE_REFERENCE.";
                SB_LOG_ERROR("Unsupported execution mode: {}", message);
                throw std::invalid_argument(message);
            }

            /**
             * @brief Records a task that has a SYCL kernel only.
             * @param family What kind of operation this is, for the diagnostic.
             * @param body Callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`,
             *        copyable and callable more than once; it orders its work behind the events
             *        it is given.
             * @throws std::invalid_argument under FORCE_REFERENCE; no task is recorded.
             */
            template<typename Body>
            TaskHandle record(const SushiRuntime::Graph::TaskMetadata& meta,
                              const OperandSet& operands, const char* family, Body&& body)
            {
                require_sycl_path(family, meta.name);
                return insert(meta, operands,
                              Internal::Detail::RecordedTask<std::decay_t<Body>>(
                                  std::forward<Body>(body), operands.retained()));
            }

            /**
             * @brief Records a task whose body runs a reference and SYCL kernel pair.
             * @param body Callable `(const Internal::KernelPairDispatcher&, sycl::queue&,
             *        const std::vector<sycl::event>&) -> sycl::event`, copyable, callable more
             *        than once, and ordering its work behind the events it is given.
             * @throws std::invalid_argument under FORCE_REFERENCE when a declared tensor is
             *         device-only USM; no task is recorded.
             */
            template<typename Body>
            TaskHandle record_pair(const SushiRuntime::Graph::TaskMetadata& meta,
                                   const OperandSet& operands, const char* family, Body&& body)
            {
                require_host_reachable(operands, family, meta.name);
                return insert(meta, operands,
                              Internal::Detail::RecordedPairTask<std::decay_t<Body>>(
                                  std::forward<Body>(body), Internal::KernelPairDispatcher(mode_),
                                  operands.retained()));
            }

        private:
            /** @brief Throws std::invalid_argument when the host kernel could not read an operand. */
            void require_host_reachable(const OperandSet& operands, const char* family,
                                        const char* name) const
            {
                if (mode_ != ExecutionMode::FORCE_REFERENCE || !operands.has_device_only_operand())
                    return;

                const std::string message = std::string(family) + " '" + name +
                                            "' cannot run under FORCE_REFERENCE: an operand is "
                                            "device-only USM, which the host reference kernel "
                                            "cannot dereference.";
                SB_LOG_ERROR("Invalid operand: {}", message);
                throw std::invalid_argument(message);
            }

            /** @brief Adds one task to the graph under a registration index drawn for it. */
            TaskHandle insert(const SushiRuntime::Graph::TaskMetadata& meta,
                              const OperandSet& operands, SushiRuntime::Graph::HostWork work)
            {
                const std::uint32_t registration = graph_->reserve_registration();
                graph_->add_task(meta, operands.read_regions(), operands.write_regions(),
                                 std::move(work), std::vector<sycl::event>{}, registration);
                return TaskHandle(identity_, registration);
            }

            /// @brief The graph every record call adds to; never null.
            SushiRuntime::Graph::TaskGraph* graph_;

            /// @brief The pool handed to operations that lease scratch; never null.
            ScratchPool* scratch_pool_;

            /// @brief The mode stamped on each task as it is recorded.
            ExecutionMode mode_;

            /// @brief The identity this recorder's handles carry.
            std::uint64_t identity_;
    };
} // namespace SushiBLAS
