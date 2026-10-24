/**************************************************************************/
/* cluster_binding.hpp                                                    */
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
 * @file cluster_binding.hpp
 * @brief Declares ClusterBinding, the cluster and the kernel registry one engine works with.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>

namespace SushiBLAS
{
    /**
     * @brief Shares packs with a cluster and hands their tasks the kernel registry.
     *
     * It exposes what record_remote() and the caller's packs use, never the whole Cluster.
     * @see modules/engine/cluster/README.md
     */
    class ClusterBinding final
    {
        public:
            /**
             * @brief Binds a cluster and the registry its tasks' kernels come from.
             * @pre kernels is not null.
             */
            ClusterBinding(
                SushiRuntime::Distributed::Cluster cluster,
                SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> kernels);

            /**
             * @brief Shares a pack's buffer with the workers until the buffer is released.
             *
             * The buffer leaves the cluster's registry when the last copy of the pack is
             * destroyed, through the allocator's release observer.
             */
            void share(const TensorPack& pack);

            /** @brief Returns whether a pack's buffer is shared with the workers. */
            [[nodiscard]] bool is_shared(const TensorPack& pack) const;

            /** @brief Returns whether a live worker advertises an operation. */
            [[nodiscard]] bool has_worker_for(SushiRuntime::OperationID operation) const;

            /** @brief Returns the registry the tasks' kernels come from. */
            [[nodiscard]] const SushiRuntime::Distributed::KernelRegistry& kernels() const noexcept
            {
                return *kernels_;
            }

        private:
            /// @brief The handle of the master the packs are shared with.
            SushiRuntime::Distributed::Cluster cluster_;

            /// @brief The registry record_remote() reads the kernels from; never null.
            SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> kernels_;
    };
} // namespace SushiBLAS
