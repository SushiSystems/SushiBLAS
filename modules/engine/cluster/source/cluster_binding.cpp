/**************************************************************************/
/* cluster_binding.cpp                                                    */
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
 * @file cluster_binding.cpp
 * @brief Defines ClusterBinding: sharing a pack and reading what the cluster supports.
 * @author Mustafa Garip
 */

#include <SushiBLAS/cluster/cluster_binding.hpp>

#include <utility>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    ClusterBinding::ClusterBinding(
        SushiRuntime::Distributed::Cluster cluster,
        SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> kernels)
        : cluster_(std::move(cluster)), kernels_(std::move(kernels))
    {
        SB_THROW_IF(!kernels_, "ClusterBinding: the kernel registry is null");
    }

    void ClusterBinding::share(const TensorPack& pack)
    {
        cluster_.share(pack.data(), pack.size_bytes());
    }

    bool ClusterBinding::is_shared(const TensorPack& pack) const
    {
        return cluster_.buffer_handle(pack.data()).has_value();
    }

    bool ClusterBinding::has_worker_for(SushiRuntime::OperationID operation) const
    {
        return cluster_.supports(operation);
    }
} // namespace SushiBLAS
