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
 * @file task_recorder.hpp
 * @brief Declares TaskRecorder, the one caller of SushiRuntime::Recorder::task in SushiBLAS.
 * @author Mustafa Garip
 */

#pragma once

#include <mutex>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <string_view>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/cost/cost_table.hpp>
#include <SushiBLAS/engine/profile_hooks.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_cost.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/launch/submitted_events.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/pair_run.hpp>
#include <SushiBLAS/engine/late_count_sink.hpp>
#include <SushiBLAS/engine/task_sink.hpp>
#include <SushiBLAS/engine/graph_task_sink.hpp>
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
             * @brief Wraps a kernel-pair body in the shape a TaskSink keeps.
             * @tparam Body Callable `(const KernelPairDispatcher&, const std::vector<sycl::event>&)`
             *         returning a `sycl::event`, or a Device::SubmittedEvents when it submits
             *         several kernels.
             */
            template<typename Body>
            class RecordedPairTask
            {
                public:
                    /** @brief Holds the body. */
                    explicit RecordedPairTask(Body body) : body_(std::make_shared<const Body>(std::move(body)))
                    {

                    }

                    /**
                     * @brief Runs the body with the dispatcher and the dependencies of the run,
                     *        and finishes the run with the event that completes after every kernel
                     *        the body submitted.
                     */
                    void operator()(PairRun& run) const
                    {
                        run.finish(runtime_events((*body_)(run.dispatcher(), run.dependencies())));
                    }

                private:
                    /// @brief The body the operation recorded; shared so the wrapper copies cheaply.
                    std::shared_ptr<const Body> body_;
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
                    sycl::event operator()(const KernelPairDispatcher&,
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
     * @brief Records tasks through one TaskSink under one execution mode, one record call at a
     *        time; the sink is a SushiRuntime::Recorder by default.
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
                : own_graph_(std::in_place, recorder), sink_(&*own_graph_),
                  late_sink_(&*own_graph_), scratch_pool_(&scratch_pool),
                  late_counts_(&late_counts), mode_(mode),
                  identity_(Internal::Detail::next_recorder_identity())
            {

            }

            /**
             * @brief Binds the recorder to a sink it records through; each pointer may be null.
             * @param sink Receives every task; it must outlive this recorder.
             * @param scratch_pool Lends scratch; scratch_pool() throws without one.
             * @param late_counts Keeps late-count storages, beside late_sink.
             * @param late_sink Takes late-count kernels; without one only ForceReference records them.
             * @throws UnsupportedExecution when the sink does not admit the mode.
             */
            explicit TaskRecorder(TaskSink& sink, ScratchPool* scratch_pool = nullptr,
                                  LateCountHold* late_counts = nullptr,
                                  ExecutionMode mode = ExecutionMode::ForceReference,
                                  LateCountSink* late_sink = nullptr)
                : sink_(&sink), late_sink_(late_sink), scratch_pool_(scratch_pool),
                  late_counts_(late_counts), mode_(mode),
                  identity_(Internal::Detail::next_recorder_identity())
            {
                require_admitted(mode);
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
             * @throws UnsupportedExecution when the sink does not admit the mode; the mode stays.
             */
            void set_mode(ExecutionMode mode)
            {
                require_admitted(mode);
                mode_ = mode;
            }

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

            /**
             * @brief Returns the pool recorded tasks lease their device scratch from.
             * @throws UnsupportedExecution when the recorder was built without one.
             */
            [[nodiscard]] ScratchPool& scratch_pool() const
            {
                if (scratch_pool_ == nullptr)
                {
                    throw UnsupportedExecution(
                        "This task recorder has no scratch pool: a recorder on a host sink is "
                        "built without one.");
                }
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
             * @param body Callable `(const KernelPairDispatcher&, deps)` returning an event, or
             *        Device::SubmittedEvents when it submits several kernels.
             * @throws UnsupportedExecution for a device-only tensor under ForceReference, and
             *         std::runtime_error when the runtime refuses; no task is recorded.
             */
            template<typename Body>
            TaskHandle record_pair(const TaskDescription& description,
                                   const OperandSet& operands, const char* family, Body&& body)
            {
                require_host_reachable(operands, family, description.name);
                return commit(
                    [&]
                    {
                        sink_->add_pair(description, operands, mode_, origin_,
                                        Internal::Detail::RecordedPairTask<std::decay_t<Body>>(
                                            std::forward<Body>(body)));
                    });
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
                if (late_sink_ == nullptr || late_counts_ == nullptr)
                {
                    throw UnsupportedExecution(
                        std::string(description.name) +
                        ": a late-count kernel is a graph node, and this recorder holds no "
                        "late-count sink.");
                }
                LateCount late = late_counts_->borrow(operands, count);
                const TaskHandle handle = commit(
                    [&]
                    {
                        late_sink_->add_late_node(
                            description, Cost::uncosted(UncostedReason::DataDependent), origin_,
                            [&](SushiRuntime::Recorder& recorder,
                                const SushiRuntime::NodeOptions& options)
                            {
                                recorder.dynamic_kernel(
                                    SushiRuntime::sized_from_device(
                                        late.words,
                                        static_cast<std::size_t>(count.storage_offset)),
                                    capacity, Internal::task_access(operands),
                                    std::forward<Kernel>(kernel), options);
                            });
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
             * @brief Runs add under the recording mutex and the profile zone, then numbers the task.
             * @param add Callable with no argument that adds exactly one task to the sink.
             * @throws std::runtime_error as the sink states; no number is taken then.
             */
            template<typename Add>
            TaskHandle commit(Add&& add)
            {
                std::uint64_t sequence = 0;
                Internal::in_recording(recording_mutex_, [&]
                {
                    // The task added to the sink and its sequence number are one step.
                    std::forward<Add>(add)();
                    sequence = next_sequence_++;
                });

                return TaskHandle(identity_, sequence);
            }

            /**
             * @brief Checks that the sink runs tasks recorded under the mode.
             * @throws UnsupportedExecution otherwise.
             */
            void require_admitted(ExecutionMode mode) const
            {
                if (!sink_->admits(mode))
                {
                    throw UnsupportedExecution(
                        "The sink of this task recorder cannot run tasks recorded under the "
                        "requested execution mode: a host sink runs ForceReference only.");
                }
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

            /// @brief The graph adapter this recorder owns when built over a SushiRuntime::Recorder.
            std::optional<GraphTaskSink> own_graph_;

            /// @brief The sink every record call adds to; never null.
            TaskSink* sink_;

            /// @brief The sink late-count kernels are added to; null when the recorder takes none.
            LateCountSink* late_sink_;

            /// @brief The pool handed to operations that lease scratch; null when there is none.
            ScratchPool* scratch_pool_;

            /// @brief The hold late-count tasks are kept in, owned by the owner of the graph; null on a sink with no graph.
            LateCountHold* late_counts_;

            /// @brief The mode stamped on each task as it is recorded.
            ExecutionMode mode_;

            /// @brief The identity this recorder's handles carry.
            std::uint64_t identity_;

            /// @brief The sequence number the next recorded task takes; guarded by recording_mutex_.
            std::uint64_t next_sequence_ = 0;

            /// @brief The origin every node recorded now carries; empty when no scope is open; guarded by recording_mutex_.
            std::string origin_;

            /// @brief Serialises every use of sink_, next_sequence_ and origin_.
            std::mutex recording_mutex_;
    };
} // namespace SushiBLAS
