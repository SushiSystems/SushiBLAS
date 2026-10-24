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

#include <mutex>
#include <atomic>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
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

            /**
             * @brief Adapts a host-only callable to the body shape record_pair() takes.
             * @tparam Host Callable `() -> void`, copyable.
             */
            template<typename Host>
            class HostOnlyBody
            {
                public:
                    /** @brief Holds the callable. */
                    explicit HostOnlyBody(Host host) : host_(std::move(host))
                    {

                    }

                    /** @brief Waits for the events it is given, then runs the callable on the host. */
                    sycl::event operator()(const KernelPairDispatcher&, sycl::queue&,
                                           const std::vector<sycl::event>& deps) const
                    {
                        sycl::event::wait(deps);
                        host_();
                        return sycl::event();
                    }

                private:
                    /// @brief The callable the task runs.
                    Host host_;
            };
        } // namespace Detail
    } // namespace Internal

    /**
     * @brief Records tasks on one SushiRuntime::Recorder under one execution mode, one record
     *        call at a time.
     * @see modules/engine/engine/README.md
     */
    class TaskRecorder final
    {
        public:
            /**
             * @brief Binds the recorder to the graph recorder it records on, the pool its tasks
             *        lease from and the hold its late-count tasks are kept in.
             * @param recorder Receives every task; it must outlive this recorder.
             * @param scratch_pool Lends operation scratch; it must outlive the recorder.
             * @param late_counts Keeps late-count storages; its owner keeps it while the graph runs.
             */
            TaskRecorder(SushiRuntime::Recorder& recorder, ScratchPool& scratch_pool,
                         LateCountHold& late_counts,
                         ExecutionMode mode = ExecutionMode::Auto) noexcept
                : recorder_(&recorder), scratch_pool_(&scratch_pool), late_counts_(&late_counts),
                  mode_(mode), identity_(Internal::Detail::next_recorder_identity())
            {

            }

            TaskRecorder(const TaskRecorder&) = delete;
            TaskRecorder& operator=(const TaskRecorder&) = delete;
            TaskRecorder(TaskRecorder&&) = delete;
            TaskRecorder& operator=(TaskRecorder&&) = delete;
            ~TaskRecorder() = default;

            /** @brief Returns the mode every later record call is recorded under. */
            [[nodiscard]] ExecutionMode mode() const noexcept { return mode_; }

            /**
             * @brief Sets the mode every later record call is recorded under.
             * @pre No thread is recording on this recorder.
             */
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
             * @param body Callable `(const KernelPairDispatcher&, sycl::queue&,
             *        const std::vector<sycl::event>&) -> sycl::event`, copyable, callable more
             *        than once, and ordering its work behind the events it is given.
             * @throws UnsupportedExecution for a device-only tensor under ForceReference, and
             *         std::runtime_error when the runtime refuses; no task is recorded.
             */
            template<typename Body>
            TaskHandle record_pair(const TaskDescription& description,
                                   const OperandSet& operands, const char* family, Body&& body)
            {
                require_host_reachable(operands, family, description.name);
                return insert(description, operands,
                              Internal::Detail::RecordedPairTask<std::decay_t<Body>>(
                                  std::forward<Body>(body), KernelPairDispatcher(mode_),
                                  operands.retained()));
            }

            /**
             * @brief Records a task over capacity indices whose width is the UINT32 or INT32 in count.
             * @param count A contiguous UINT32 or INT32 tensor in Residency::Host that a previous node writes.
             * @param host_body Callable `() -> void` that runs the indices under ForceReference.
             * @param kernel Device-copyable callable `(std::size_t i)` for every other mode.
             * @note Outside ForceReference the hold given at construction keeps its storages;
             *       see modules/engine/engine/README.md.
             */
            template<typename HostBody, typename Kernel>
            TaskHandle record_late_pair(const TaskDescription& description,
                                        const OperandSet& operands, const char* family,
                                        const Tensor& count, std::size_t capacity,
                                        HostBody&& host_body, Kernel&& kernel)
            {
                require_late_count(count, description.name);
                if (mode_ == ExecutionMode::ForceReference)
                {
                    return record_pair(
                        description, operands, family,
                        Internal::Detail::HostOnlyBody<std::decay_t<HostBody>>(
                            std::forward<HostBody>(host_body)));
                }

                require_host_reachable(operands, family, description.name);
                LateCount late = late_counts_->borrow(operands, count);
                const TaskHandle handle = insert(
                    description,
                    [&](SushiRuntime::Recorder& recorder, const SushiRuntime::NodeOptions& options)
                    {
                        recorder.dynamic_kernel(
                            SushiRuntime::sized_from_device(
                                late.words, static_cast<std::size_t>(count.storage_offset)),
                            capacity,
                            access_of(operands), std::forward<Kernel>(kernel), options);
                    });
                late_counts_->keep(std::move(late.owner));
                return handle;
            }

            /**
             * @brief Runs work while no record call of this recorder adds a task, and returns
             *        what it returns.
             * @param work Callable with no argument; it must not record on this recorder.
             * @see modules/engine/engine/README.md
             */
            template<typename Work>
            decltype(auto) run_exclusive(Work&& work)
            {
                const std::lock_guard<std::mutex> lock(recording_mutex_);
                return std::forward<Work>(work)();
            }

        private:
            /**
             * @brief Builds the access of operands: its regions, and the owners of borrowed
             *        storage a run checks first.
             */
            [[nodiscard]] static SushiRuntime::Access access_of(const OperandSet& operands)
            {
                SushiRuntime::Access access(operands.read_regions(), operands.write_regions());
                access.lives = operands.owner_liveness();
                return access;
            }

            /**
             * @brief Records one task under the name, the operation and the parameters of
             *        description.
             * @param work Callable `(sycl::queue&, const std::vector<sycl::event>&)` that returns
             *        the event ending its work.
             * @throws std::runtime_error with the runtime's message when the graph recorder
             *         refuses the recording; the next run returns the same refusal.
             */
            template<typename Work>
            TaskHandle insert(const TaskDescription& description,
                              const OperandSet& operands, Work&& work)
            {
                return insert(
                    description,
                    [&](SushiRuntime::Recorder& recorder, const SushiRuntime::NodeOptions& options)
                    {
                        recorder.task(access_of(operands), std::forward<Work>(work), options);
                    });
            }

            /**
             * @brief Records one node under the name, the operation, the determinism and the
             *        parameters of description.
             * @param record Callable `(SushiRuntime::Recorder&, const SushiRuntime::NodeOptions&)`
             *        that adds exactly one node.
             * @throws std::runtime_error as the overload above.
             */
            template<typename Record>
            TaskHandle insert(const TaskDescription& description, Record&& record)
            {
                static_assert(TaskDescription::K_MAX_PARAMETERS <=
                                  SushiRuntime::NodeOptions::K_MAX_PARAMETERS,
                              "A task description carries more parameters than a runtime node "
                              "takes.");
                SushiRuntime::NodeOptions options;
                options.name = description.name;
                options.operation = description.operation;
                options.determinism = description.determinism;
                options.parameters = SushiRuntime::span<const std::uint64_t>(
                    description.parameters, TaskDescription::K_MAX_PARAMETERS);

                // The count before, the recording and the sequence number are one step.
                const std::lock_guard<std::mutex> lock(recording_mutex_);
                const std::size_t recorded_before = recorder_->node_count();
                std::forward<Record>(record)(*recorder_, options);
                if (recorder_->node_count() == recorded_before)
                {
                    const SushiRuntime::Error* refusal = recorder_->refusal();
                    SB_THROW_IF(true, "{}: the runtime refused the recording: {}",
                                description.name,
                                refusal != nullptr ? refusal->message.c_str() : "no reason kept");
                }

                return TaskHandle(identity_, next_sequence_++);
            }

            /**
             * @brief Checks the tensor a late width is read from.
             * @throws std::invalid_argument for a count that is not a contiguous UINT32 or INT32 tensor of
             *         at least one element with storage in Residency::Host.
             */
            static void require_late_count(const Tensor& count, const char* name)
            {
                Dispatch::require_min_elements(count, 1, name, "count");
                Dispatch::require_storage(count, name, "count");
                Dispatch::require_contiguous(count, name, "count");
                Dispatch::require_operand(
                    count.dtype == Core::DataType::UInt32 || count.dtype == Core::DataType::Int32,
                    name, "count", "has dtype {}, expected UINT32 or INT32",
                    Core::to_string(count.dtype));
                Dispatch::require_operand(
                    count.storage->residency() == SushiRuntime::Residency::Host, name, "count",
                    "lives in {} memory; the runtime reads a late width from Residency::Host",
                    SushiRuntime::residency_name(count.storage->residency()));
            }

            /// @brief The graph recorder every record call adds to; never null.
            SushiRuntime::Recorder* recorder_;

            /// @brief The pool handed to operations that lease scratch; never null.
            ScratchPool* scratch_pool_;

            /// @brief The hold late-count tasks are kept in, owned by the graph's owner; never null.
            LateCountHold* late_counts_;

            /// @brief The mode stamped on each task as it is recorded.
            ExecutionMode mode_;

            /// @brief The identity this recorder's handles carry.
            std::uint64_t identity_;

            /// @brief The sequence number the next recorded task takes; guarded by recording_mutex_.
            std::uint64_t next_sequence_ = 0;

            /// @brief Serialises every use of recorder_ and next_sequence_.
            std::mutex recording_mutex_;
    };
} // namespace SushiBLAS
