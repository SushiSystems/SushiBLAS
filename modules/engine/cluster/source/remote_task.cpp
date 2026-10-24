/**************************************************************************/
/* remote_task.cpp                                                        */
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
 * @file remote_task.cpp
 * @brief Defines record_remote(), bind_views() and the fingerprinted operation name.
 * @author Mustafa Garip
 */

#include <SushiBLAS/cluster/remote_task.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/distributed/registered_task.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Holds the hex digits of a fingerprint, lower case. */
        constexpr char K_HEX_DIGITS[] = "0123456789abcdef";

        /** @brief Appends the view of each pack, after checking it is shared. */
        void append_views(std::vector<SushiRuntime::Distributed::BufferView>& views,
                          const ClusterBinding& binding,
                          SushiRuntime::span<const TensorPack* const> packs,
                          const std::string& name)
        {
            for (const TensorPack* pack : packs)
            {
                SB_THROW_IF(pack == nullptr, "record_remote({}): a pack is null", name);
                SB_THROW_IF(!binding.is_shared(*pack),
                            "record_remote({}): a pack is not shared with the cluster", name);
                views.push_back(pack->view());
            }
        }
    } // namespace

    std::string remote_operation_name(const std::string& name, std::uint64_t fingerprint)
    {
        std::string full = name;
        full.reserve(name.size() + 17);
        full.push_back(':');
        for (int shift = 60; shift >= 0; shift -= 4)
            full.push_back(K_HEX_DIGITS[(fingerprint >> shift) & 0xfU]);
        return full;
    }

    SushiRuntime::OperationID remote_operation(const std::string& name, std::uint64_t fingerprint)
    {
        const std::string full = remote_operation_name(name, fingerprint);
        return SushiRuntime::OperationID(SushiRuntime::OperationID::hash(full.data(), full.size()));
    }

    std::size_t record_remote(SushiRuntime::Recorder& recorder, const ClusterBinding& binding,
                              const SushiRuntime::NodeOptions& options,
                              SushiRuntime::span<const TensorPack* const> inputs,
                              SushiRuntime::span<const TensorPack* const> outputs)
    {
        const std::string name =
            options.name.empty() ? std::string("unnamed") : std::string(options.name);
        SB_THROW_IF(!binding.kernels().has(options.operation),
                    "record_remote({}): the registry has no kernel for the operation", name);

        std::vector<SushiRuntime::Distributed::BufferView> views;
        views.reserve(inputs.size() + outputs.size());
        append_views(views, binding, inputs, name);
        append_views(views, binding, outputs, name);

        const std::size_t recorded_before = recorder.rule_count();
        const std::size_t position = SushiRuntime::Distributed::record_registered_task(
            recorder, binding.kernels(), options,
            SushiRuntime::span<const SushiRuntime::Distributed::BufferView>(views.data(),
                                                                            inputs.size()),
            SushiRuntime::span<const SushiRuntime::Distributed::BufferView>(
                views.data() + inputs.size(), outputs.size()));
        if (recorder.rule_count() == recorded_before)
        {
            const SushiRuntime::Error* refusal = recorder.refusal();
            SB_THROW_IF(true, "record_remote({}): the runtime refused the recording: {}", name,
                        refusal != nullptr ? refusal->message.c_str() : "no reason kept");
        }
        return position;
    }

    std::vector<BoundPack> bind_views(
        SushiRuntime::span<const SushiRuntime::Distributed::BufferView> views,
        SushiRuntime::span<const PackLayout> layouts)
    {
        SB_THROW_IF(views.size() != layouts.size(),
                    "bind_views: {} views were received and {} layouts were given", views.size(),
                    layouts.size());

        std::vector<BoundPack> bound;
        bound.reserve(views.size());
        for (std::size_t index = 0; index < views.size(); ++index)
        {
            const SushiRuntime::Distributed::BufferView& view = views[index];
            const PackLayout& layout = layouts[index];
            SB_THROW_IF(view.data == nullptr, "bind_views: view {} names no memory", index);
            SB_THROW_IF(view.size_bytes != layout.size_bytes(),
                        "bind_views: view {} holds {} bytes and its layout {} bytes", index,
                        view.size_bytes, layout.size_bytes());
            bound.emplace_back(view.data, layout);
        }
        return bound;
    }
} // namespace SushiBLAS
