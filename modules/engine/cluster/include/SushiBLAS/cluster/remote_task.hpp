/**************************************************************************/
/* remote_task.hpp                                                        */
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
 * @file remote_task.hpp
 * @brief Declares record_remote() and bind_views(), a registered task over packs.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/distributed/buffer_view.hpp>

namespace SushiBLAS
{
    /**
     * @brief Returns the name "<name>:<fingerprint as 16 lower-case hex digits>".
     *
     * A worker registers its kernel under this name, so a worker whose layout differs
     * advertises another OperationID and the master never sends it the task.
     */
    [[nodiscard]] std::string remote_operation_name(const std::string& name,
                                                    std::uint64_t fingerprint);

    /** @brief Returns the OperationID of remote_operation_name(name, fingerprint). */
    [[nodiscard]] SushiRuntime::OperationID remote_operation(const std::string& name,
                                                             std::uint64_t fingerprint);

    /**
     * @brief Records a registered task over packs, one view per pack, into the caller's recorder.
     * @pre Every pack is shared through binding and outlives the run of the graph.
     * @throws std::runtime_error on a null or unshared pack, an operation the registry has
     *         no kernel for, or a recording the runtime refuses.
     * @see modules/engine/cluster/README.md
     */
    std::size_t record_remote(SushiRuntime::Recorder& recorder, const ClusterBinding& binding,
                              const SushiRuntime::NodeOptions& options,
                              SushiRuntime::span<const TensorPack* const> inputs,
                              SushiRuntime::span<const TensorPack* const> outputs);

    /**
     * @brief Checks each view a kernel received against a layout and binds its segments.
     * @param layouts One layout per view, in the views' order; they outlive the result.
     * @throws std::runtime_error when the counts differ or a view's size is not its layout's.
     */
    [[nodiscard]] std::vector<BoundPack> bind_views(
        SushiRuntime::span<const SushiRuntime::Distributed::BufferView> views,
        SushiRuntime::span<const PackLayout> layouts);
} // namespace SushiBLAS
