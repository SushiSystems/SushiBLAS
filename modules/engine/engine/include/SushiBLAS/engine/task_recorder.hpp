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
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <string_view>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/cost/cost_table.hpp>
#include <SushiBLAS/engine/profile_hooks.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/launch/submitted_events.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/api/graph/dispatch_events.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        namespace Detail
        {
            /**
             * @brief Returns the runtime's name for the ceiling class of a real dtype.
             * @return F16 or F32; Unstated for a type the runtime names no class for.
             */
            [[nodiscard]] constexpr SushiRuntime::NodeDType node_dtype_of(Core::DataType dtype) noexcept
            {
                switch (dtype)
                {
                    case Core::DataType::Half:
                        return SushiRuntime::NodeDType::F16;
                    case Core::DataType::Float32:
                        return SushiRuntime::NodeDType::F32;
                    default:
                        return SushiRuntime::NodeDType::Unstated;
                }
            }

            /** @brief Returns the event of a one-kernel body as it is. */
            [[nodiscard]] inline sycl::event runtime_events(sycl::event event) noexcept
            {
                return event;
            }

            /**
             * @brief Tells the runtime about every kernel a body submitted, and returns the last.
             *
             * The runtime then times the dispatch from the first kernel's start to the last
             * kernel's end. Every kernel is ordered behind the one before it, so the last event
             * completes after all of them.
             */
            [[nodiscard]] inline sycl::event runtime_events(const Device::SubmittedEvents& events)
            {
                for (const sycl::event& event : events.events())
                    SushiRuntime::note_dispatch_event(event);
                return events;
            }

            /** @brief Returns an identity no other TaskRecorder of this process holds; never 0. */
            [[nodiscard]] inline std::uint64_t next_recorder_identity() noexcept
            {
                static std::atomic<std::uint64_t> next{1};
                return next.fetch_add(1, std::memory_order_relaxed);
            }

            /**
             * @brief Runs a kernel-pair body with its dispatcher and keeps its operands' storage.
             * @tparam Body Callable `(const KernelPairDispatcher&, sycl::queue&,
             *         const std::vector<sycl::event>&)` returning a `sycl::event`, or a
             *         Device::SubmittedEvents when it submits several kernels.
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

                    /**
                     * @brief Runs the body with the dispatcher fixed at record time.
                     * @return The event that completes after every kernel the body submitted.
                     */
                    sycl::event operator()(sycl::queue& q, const std::vector<sycl::event>& deps) const
                    {
                        return runtime_events(body_(dispatcher_, q, deps));
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
             * @brief Keeps a label open on a recorder: every node recorded while it lives carries it
             *        as its origin.
             *
             * A scope opened inside another restores the outer label when it ends. A recorder
             * that never opens a scope records every node with an empty origin.
             * @see modules/engine/engine/README.md
             */
            class OriginScope final
            {
                public:
                    OriginScope(const OriginScope&) = delete;
                    OriginScope& operator=(const OriginScope&) = delete;
                    OriginScope& operator=(OriginScope&&) = delete;

                    /** @brief Takes over the scope of @p other, which then restores nothing. */
                    OriginScope(OriginScope&& other) noexcept
                        : recorder_(std::exchange(other.recorder_, nullptr)),
                          previous_(std::move(other.previous_))
                    {

                    }

                    /** @brief Restores the label that was current when the scope opened. */
                    ~OriginScope()
                    {
                        if (recorder_ != nullptr)
                            recorder_->set_origin(std::move(previous_));
                    }

                private:
                    friend class TaskRecorder;

                    /** @brief Holds the recorder to restore and the label it had. */
                    OriginScope(TaskRecorder& recorder, std::string previous) noexcept
                        : recorder_(&recorder), previous_(std::move(previous))
                    {

                    }

                    /// @brief The recorder whose label the scope restores; null after a move.
                    TaskRecorder* recorder_;

                    /// @brief The label the recorder had before the scope opened.
                    std::string previous_;
            };

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

            /**
             * @brief Opens a scope in which every recorded node carries @p label as its origin.
             * @pre No thread is inside run_exclusive() of this recorder.
             * @return A scope to keep for as long as the label applies.
             */
            [[nodiscard]] OriginScope origin(std::string_view label)
            {
                const std::lock_guard<std::mutex> lock(recording_mutex_);
                std::string previous = std::move(origin_);
                origin_.assign(label);
                return OriginScope(*this, std::move(previous));
            }

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
             * @param body Callable `(const KernelPairDispatcher&, sycl::queue&, const
             *        std::vector<sycl::event>&)` returning a `sycl::event`, or SubmittedEvents.
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
                const TaskHandle handle = insert_node(
                    description, Cost::uncosted(UncostedReason::DataDependent),
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
             * @brief Records one task under the name, operation and parameters of description.
             * @param work Callable `(sycl::queue&, const std::vector<sycl::event>&)` returning the
             *        event that ends its work.
             * @note The cost is the table's cost for the operation and shape plus the bytes.
             * @throws std::runtime_error with the runtime's message when the recorder refuses.
             */
            template<typename Work>
            TaskHandle insert(const TaskDescription& description,
                              const OperandSet& operands, Work&& work)
            {
                return insert_node(
                    description,
                    cost_of(description.operation, description.shape)
                        .with_bytes(operands.declared_bytes()),
                    [&](SushiRuntime::Recorder& recorder, const SushiRuntime::NodeOptions& options)
                    {
                        recorder.task(access_of(operands), std::forward<Work>(work), options);
                    });
            }

            /**
             * @brief Records one node under the name, the operation, the determinism, the
             *        parameters and the cost.
             * @param cost What the node states about its work; an uncosted cost states none.
             * @param record Callable `(SushiRuntime::Recorder&, const SushiRuntime::NodeOptions&)`
             *        that adds exactly one node.
             * @throws std::runtime_error as the overload above.
             */
            template<typename Record>
            TaskHandle insert_node(const TaskDescription& description, const Cost& cost,
                                   Record&& record)
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
                if (cost.is_costed())
                {
                    SushiRuntime::NodeCost node_cost;
                    node_cost.flops = cost.flops;
                    node_cost.bytes = cost.bytes;
                    node_cost.dtype = node_dtype_of(cost.dtype);
                    options.cost = node_cost;
                }

                std::uint64_t sequence = 0;
                Internal::in_recording(recording_mutex_, [&]
                {
                    // The count before, the recording and the sequence number are one step.
                    options.origin = origin_;
                    const std::size_t recorded_before = recorder_->node_count();
                    std::forward<Record>(record)(*recorder_, options);
                    if (recorder_->node_count() == recorded_before)
                    {
                        const SushiRuntime::Error* refusal = recorder_->refusal();
                        SB_THROW_IF(true, "{}: the runtime refused the recording: {}",
                                    description.name,
                                    refusal != nullptr ? refusal->message.c_str()
                                                       : "no reason kept");
                    }
                    sequence = next_sequence_++;
                });

                return TaskHandle(identity_, sequence);
            }

            /** @brief Sets the label recorded nodes carry; the end of an OriginScope calls it. */
            void set_origin(std::string label)
            {
                const std::lock_guard<std::mutex> lock(recording_mutex_);
                origin_ = std::move(label);
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

            /// @brief The origin every node recorded now carries; empty when no scope is open; guarded by recording_mutex_.
            std::string origin_;

            /// @brief Serialises every use of recorder_, next_sequence_ and origin_.
            std::mutex recording_mutex_;
    };
} // namespace SushiBLAS
