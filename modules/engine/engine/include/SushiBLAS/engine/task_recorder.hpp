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
 * @brief Declares TaskRecorder, the one caller of SushiRuntime::Recorder::task in SushiBLAS.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/operation_id.hpp>

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
     * @brief Records tasks on one SushiRuntime::Recorder under one execution mode.
     * @see modules/engine/engine/README.md
     */
    class TaskRecorder final
    {
        public:
            /**
             * @brief Binds the recorder to the graph recorder it records on and the pool its tasks
             *        lease from.
             * @param recorder Receives every task; it must outlive this recorder.
             * @param scratch_pool Lends operation scratch; it must outlive the recorder.
             */
            TaskRecorder(SushiRuntime::Recorder& recorder, ScratchPool& scratch_pool,
                         ExecutionMode mode = ExecutionMode::Auto) noexcept
                : recorder_(&recorder), scratch_pool_(&scratch_pool), mode_(mode),
                  identity_(Internal::Detail::next_recorder_identity())
            {

            }

            TaskRecorder(const TaskRecorder&) = delete;
            TaskRecorder& operator=(const TaskRecorder&) = delete;
            TaskRecorder(TaskRecorder&&) = delete;
            TaskRecorder& operator=(TaskRecorder&&) = delete;
            ~TaskRecorder() = default;

            /** @brief Returns the mode every later record call is recorded under. */
            [[nodiscard]] ExecutionMode mode() const noexcept { return mode_; }

            /** @brief Sets the mode every later record call is recorded under. */
            void set_mode(ExecutionMode mode) noexcept { mode_ = mode; }

            /** @brief Returns the pool recorded tasks lease their device scratch from. */
            [[nodiscard]] ScratchPool& scratch_pool() const noexcept
            {
                return *scratch_pool_;
            }

            /**
             * @brief Throws as record_pair() would, for a caller that must refuse before a side effect.
             * @throws UnsupportedExecution under ForceReference when a declared tensor is
             *         device-only USM.
             */
            void require_host_reachable(const OperandSet& operands, const char* family,
                                        const char* name) const
            {
                if (mode_ != ExecutionMode::ForceReference || !operands.has_device_only_operand())
                    return;

                Dispatch::Detail::throw_execution_error(
                    name,
                    "{} cannot run under ForceReference: an operand is device-only USM, which "
                    "the host reference kernel cannot dereference",
                    family);
            }

            /**
             * @brief Records a task whose body runs a reference and SYCL kernel pair.
             * @param body Callable `(const Internal::KernelPairDispatcher&, sycl::queue&,
             *        const std::vector<sycl::event>&) -> sycl::event`, copyable, callable more
             *        than once, and ordering its work behind the events it is given.
             * @throws UnsupportedExecution under ForceReference when a declared tensor is
             *         device-only USM; no task is recorded.
             */
            template<typename Body>
            TaskHandle record_pair(const SushiRuntime::Internal::Graph::TaskMetadata& meta,
                                   const OperandSet& operands, const char* family, Body&& body)
            {
                require_host_reachable(operands, family, meta.name);
                return insert(meta, operands,
                              Internal::Detail::RecordedPairTask<std::decay_t<Body>>(
                                  std::forward<Body>(body), Internal::KernelPairDispatcher(mode_),
                                  operands.retained()));
            }

        private:
            /**
             * @brief Records one task under the name, the operation and the parameters of meta.
             * @param work Callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`.
             * @return The handle of the task, or the handle of no task when the graph recorder
             *         refused the recording; the refusal is what the next run returns.
             */
            template<typename Work>
            TaskHandle insert(const SushiRuntime::Internal::Graph::TaskMetadata& meta,
                              const OperandSet& operands, Work&& work)
            {
                SushiRuntime::NodeOptions options;
                options.name = meta.name;
                options.operation = meta.op_id;
                options.parameters = SushiRuntime::span<const std::uint64_t>(
                    meta.params, SushiRuntime::Internal::Graph::TaskMetadata::K_MAX_PARAMS);

                const std::size_t recorded_before = recorder_->node_count();
                recorder_->task(
                    SushiRuntime::Access(operands.read_regions(), operands.write_regions()),
                    std::forward<Work>(work), options);
                if (recorder_->node_count() == recorded_before)
                    return TaskHandle();

                return TaskHandle(identity_, next_sequence_++);
            }

            /// @brief The graph recorder every record call adds to; never null.
            SushiRuntime::Recorder* recorder_;

            /// @brief The pool handed to operations that lease scratch; never null.
            ScratchPool* scratch_pool_;

            /// @brief The mode stamped on each task as it is recorded.
            ExecutionMode mode_;

            /// @brief The identity this recorder's handles carry.
            std::uint64_t identity_;

            /// @brief The sequence number the next recorded task takes.
            std::uint64_t next_sequence_ = 0;
    };
} // namespace SushiBLAS
