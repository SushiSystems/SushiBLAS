/**************************************************************************/
/* host_primitive_recorder.hpp                                            */
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
 * @file host_primitive_recorder.hpp
 * @brief Declares HostPrimitiveRecorder, the in-memory PrimitiveRecorder adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/device_sized.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/primitive_recorder.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Keeps a record of each node a primitive records and runs their host forms.
         *
         * Scratch comes from memory this recorder owns. run() executes the nodes in record
         * order, so a resolve reads the live counts of that run before the nodes after it.
         * @see modules/foundation/graph_primitives/README.md
         */
        class HostPrimitiveRecorder
        {
            public:
                /** @brief Marks this type as a PrimitiveRecorder. */
                static constexpr bool K_PRIMITIVE_RECORDER = true;

                /** @brief Guard of one rule; nodes recorded while it lives share the rule. */
                class RuleScope
                {
                    public:
                        RuleScope(RuleScope&& other) noexcept : recorder_(other.recorder_)
                        {
                            other.recorder_ = nullptr;
                        }

                        RuleScope(const RuleScope&) = delete;
                        RuleScope& operator=(const RuleScope&) = delete;
                        RuleScope& operator=(RuleScope&&) = delete;

                        ~RuleScope()
                        {
                            if (recorder_ != nullptr)
                                recorder_->rule_open_ = false;
                        }

                    private:
                        friend class HostPrimitiveRecorder;

                        explicit RuleScope(HostPrimitiveRecorder* recorder) noexcept
                            : recorder_(recorder)
                        {
                        }

                        HostPrimitiveRecorder* recorder_ = nullptr;
                };

                /** @brief Opens a rule; a rule opened inside another joins it. */
                [[nodiscard]] RuleScope rule() noexcept
                {
                    if (rule_open_)
                        return RuleScope(nullptr);
                    rule_open_ = true;
                    rule_has_node_ = false;
                    return RuleScope(this);
                }

                /** @brief Allocates a zeroed block of count elements this recorder owns. */
                template <typename T>
                [[nodiscard]] SushiRuntime::Result<T*>
                scratch(std::size_t count, SushiRuntime::DeviceIndex, SushiRuntime::Residency)
                {
                    constexpr std::size_t unit = sizeof(std::max_align_t);
                    const std::size_t units = (count * sizeof(T) + unit - 1) / unit;
                    blocks_.emplace_back(units == 0 ? 1 : units);
                    return reinterpret_cast<T*>(blocks_.back().data());
                }

                /** @brief Records the task that resolves counts; run() calls fill with them. */
                template <std::size_t K, class Fill>
                std::size_t resolve(const std::array<SushiRuntime::LateCount, K>& counts,
                                    SushiRuntime::DeviceIndex, SushiRuntime::Access access,
                                    Fill fill, SushiRuntime::NodeOptions options = {})
                {
                    PrimitiveNodeRecord node = describe(PrimitiveNodeKind::Resolve, access,
                                                        options);
                    node.capacity = K;
                    for (const SushiRuntime::LateCount& count : counts)
                        node.late_capacities.push_back(count.capacity());
                    forms_.push_back([counts, fill](Fault& fault)
                    {
                        std::array<std::size_t, K> live{};
                        for (std::size_t i = 0; i < K; ++i)
                            live[i] = live_count_of(counts[i], fault);
                        fill(live);
                    });
                    return keep(std::move(node));
                }

                /** @brief Records a launch of count work-items. */
                template <class F>
                std::size_t kernel(std::size_t count, SushiRuntime::Access access, F&& body,
                                   SushiRuntime::NodeOptions options = {})
                {
                    return kernel(SushiRuntime::DeviceSized{count, nullptr, nullptr},
                                  std::move(access), std::forward<F>(body), options);
                }

                /** @brief Records a launch of a device-sized capacity. */
                template <class F>
                std::size_t kernel(SushiRuntime::DeviceSized launch, SushiRuntime::Access access,
                                   F&& body, SushiRuntime::NodeOptions options = {})
                {
                    PrimitiveNodeRecord node = describe(PrimitiveNodeKind::Kernel, access,
                                                        options);
                    node.capacity = launch.capacity;
                    node.has_live_width = (launch.live_width != nullptr);
                    node.has_live_words = (launch.live_words != nullptr);
                    std::decay_t<F> kept(std::forward<F>(body));
                    forms_.push_back([kept, launch](Fault&)
                    {
                        for (std::size_t t = 0; t < launch.capacity; ++t)
                            run_item(kept, launch, t);
                    });
                    return keep(std::move(node));
                }

                /** @brief Runs every recorded node in record order. */
                [[nodiscard]] SushiRuntime::Result<void> run()
                {
                    Fault fault;
                    for (const auto& form : forms_)
                        form(fault);
                    if (fault.raised)
                        return SushiRuntime::make_error(SushiRuntime::Error{
                            SushiRuntime::ErrorCode::KernelFailed,
                            "a late count exceeded its capacity: " +
                                std::to_string(fault.live) + " over " +
                                std::to_string(fault.capacity)});
                    return {};
                }

                /** @brief A host recorder refuses nothing; the pointer is always null. */
                [[nodiscard]] const SushiRuntime::Error* refusal() const noexcept
                {
                    return nullptr;
                }

                /** @brief Returns the record of each node, in record order. */
                [[nodiscard]] const std::vector<PrimitiveNodeRecord>& nodes() const noexcept
                {
                    return nodes_;
                }

                /** @brief Returns how many rules the recorded nodes belong to. */
                [[nodiscard]] std::size_t rule_count() const noexcept { return rule_count_; }

                /**
                 * @brief Returns how many nodes read or write a region overlapping region.
                 *
                 * A node recorded later that writes region waits for exactly these nodes.
                 */
                [[nodiscard]] std::size_t
                nodes_touching(const SushiRuntime::BufferRegion& region) const
                {
                    std::size_t touching = 0;
                    for (const PrimitiveNodeRecord& node : nodes_)
                        if (touches(node.reads, region) || touches(node.writes, region))
                            ++touching;
                    return touching;
                }

            private:
                /** @brief Returns whether any of regions overlaps region. */
                static bool touches(const std::vector<SushiRuntime::BufferRegion>& regions,
                                    const SushiRuntime::BufferRegion& region)
                {
                    for (const SushiRuntime::BufferRegion& candidate : regions)
                        if (candidate.overlaps(region))
                            return true;
                    return false;
                }

                /** @brief Holds the first late count that exceeded its capacity in a run. */
                struct Fault
                {
                    bool raised = false;       /**< Whether a count exceeded its capacity. */
                    std::size_t live = 0;      /**< The live count that did. */
                    std::size_t capacity = 0;  /**< The capacity it exceeded. */
                };

                /** @brief Reads the live value of count, as the runtime's resolver does. */
                static std::size_t live_count_of(const SushiRuntime::LateCount& count,
                                                 Fault& fault)
                {
                    std::size_t live = count.capacity();
                    if (const std::uint32_t* slot = SushiRuntime::Detail::LateCountAccess::slot(count))
                        live = static_cast<std::size_t>(*slot);
                    else if (const auto* host = SushiRuntime::Detail::LateCountAccess::host(count))
                        live = (*host)();
                    if (live <= count.capacity())
                        return live;
                    if (!fault.raised)
                        fault = Fault{true, live, count.capacity()};
                    return 0;
                }

                /** @brief Runs item t of a launch, passing as many live words as the body takes. */
                template <class F>
                static void run_item(const F& body, const SushiRuntime::DeviceSized& launch,
                                     std::size_t t)
                {
                    if (launch.live_width != nullptr &&
                        t >= static_cast<std::size_t>(*launch.live_width))
                        return;
                    if constexpr (std::is_invocable_v<const F&, std::size_t, std::size_t,
                                                      std::size_t>)
                        body(t, static_cast<std::size_t>(launch.live_words[0]),
                             static_cast<std::size_t>(launch.live_words[1]));
                    else if constexpr (std::is_invocable_v<const F&, std::size_t, std::size_t>)
                        body(t, static_cast<std::size_t>(launch.live_words[0]));
                    else
                        body(t);
                }

                /** @brief Builds the record of a node from its access and options. */
                PrimitiveNodeRecord describe(PrimitiveNodeKind kind,
                                             const SushiRuntime::Access& access,
                                             const SushiRuntime::NodeOptions& options)
                {
                    PrimitiveNodeRecord node;
                    node.kind = kind;
                    if (rule_open_)
                    {
                        if (!rule_has_node_)
                        {
                            rule_has_node_ = true;
                            ++rule_count_;
                        }
                    }
                    else
                    {
                        ++rule_count_;
                    }
                    node.rule = rule_count_ - 1;
                    node.name = std::string(options.name);
                    node.determinism = options.determinism;
                    node.operation = options.operation;
                    node.width = options.width;
                    node.reads = access.reads;
                    node.writes = access.writes;
                    return node;
                }

                /** @brief Stores a record and returns its place. */
                std::size_t keep(PrimitiveNodeRecord node)
                {
                    nodes_.push_back(std::move(node));
                    return nodes_.size() - 1;
                }

                /** @brief The record of each node, in record order. */
                std::vector<PrimitiveNodeRecord> nodes_;

                /** @brief The host form of each node, in record order. */
                std::vector<std::function<void(Fault&)>> forms_;

                /** @brief The scratch blocks this recorder owns, in max-aligned units. */
                std::vector<std::vector<std::max_align_t>> blocks_;

                /** @brief The number of rules opened by a recorded node. */
                std::size_t rule_count_ = 0;

                /** @brief Whether a rule guard is alive. */
                bool rule_open_ = false;

                /** @brief Whether the open rule holds a node yet. */
                bool rule_has_node_ = false;
        };

    } // namespace GraphPrimitives
} // namespace SushiBLAS
