/**************************************************************************/
/* test_cluster_binding.cpp                                               */
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
 * @file test_cluster_binding.cpp
 * @brief Tests that ClusterBinding shares a pack with a cluster and reports what it supports.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/distributed.hpp>
#include <SushiRuntime/distributed/in_process_transport.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using SushiRuntime::Distributed::Cluster;
            using SushiRuntime::Distributed::InProcessFabric;
            using SushiRuntime::Distributed::InProcessTransport;
            using SushiRuntime::Distributed::KernelRegistry;

            /** @brief Returns the layout of one float tensor of 16 elements. */
            PackLayout sixteen_floats()
            {
                const std::int64_t dims[] = {16};
                const std::vector<Tensor> tensors = {
                    Tensor(SushiRuntime::sushi_ptr<Storage>(),
                           SushiRuntime::span<const std::int64_t>(dims, 1),
                           Core::DataType::Float32)};
                return PackLayout::of(
                    SushiRuntime::span<const Tensor>(tensors.data(), tensors.size()));
            }
        } // namespace

        /** @brief Tests that a shared pack is known to the cluster and an unshared one is not. */
        TEST(Medium_ClusterBinding, SharesAPackWithTheCluster)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            InProcessFabric fabric;
            Cluster cluster =
                Cluster::attach(runtime, InProcessTransport::make_master(fabric)).value();
            ClusterBinding binding(cluster, SushiRuntime::make_sushi<KernelRegistry>());

            const TensorPack shared(runtime, sixteen_floats());
            const TensorPack unshared(runtime, sixteen_floats());
            binding.share(shared);

            EXPECT_TRUE(binding.is_shared(shared));
            EXPECT_FALSE(binding.is_shared(unshared));
            EXPECT_FALSE(binding.has_worker_for(SushiRuntime::OperationID(1)));
        }

        /** @brief Tests that a binding refuses a null kernel registry. */
        TEST(Medium_ClusterBinding, RefusesANullRegistry)
        {
            if (sycl::device::get_devices().empty())
                GTEST_SKIP() << "SYCL reports no device.";

            SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
            InProcessFabric fabric;
            Cluster cluster =
                Cluster::attach(runtime, InProcessTransport::make_master(fabric)).value();

            EXPECT_THROW(ClusterBinding(cluster, SushiRuntime::sushi_ptr<KernelRegistry>()),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
