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
 * @file test_cluster_binding.cpp
 * @brief Tests that ClusterBinding shares a pack with a cluster and reports what it supports.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <optional>
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

        /**
         * @brief Holds the runtime and the master cluster every ClusterBinding test starts from.
         *
         * The runtime is a fixture member because Cluster::attach() changes the runtime it
         * attaches to, so the shared test runtime cannot serve.
         */
        class Medium_ClusterBinding : public ::testing::Test
        {
            protected:
                void SetUp() override
                {
                    if (sycl::device::get_devices().empty())
                        GTEST_SKIP() << "SYCL reports no device.";

                    runtime_.emplace(SushiRuntime::Runtime::create().value());
                    cluster_.emplace(
                        Cluster::attach(*runtime_, InProcessTransport::make_master(fabric_))
                            .value());
                }

                /** @brief Returns a binding over the master cluster and an empty registry. */
                ClusterBinding make_binding()
                {
                    return ClusterBinding(*cluster_, SushiRuntime::make_sushi<KernelRegistry>());
                }

                /** @brief The runtime the cluster attached to; empty when no device exists. */
                std::optional<SushiRuntime::Runtime> runtime_;

                /** @brief The fabric the master transport joins. */
                InProcessFabric fabric_;

                /** @brief The master cluster; empty when no device exists. */
                std::optional<Cluster> cluster_;
        };

        /** @brief Tests that a pack handed to share() is reported shared. */
        TEST_F(Medium_ClusterBinding, ASharedPackIsShared)
        {
            ClusterBinding binding = make_binding();
            const TensorPack shared(*runtime_, sixteen_floats(), 0);

            binding.share(shared);

            EXPECT_TRUE(binding.is_shared(shared));
        }

        /** @brief Tests that a pack never handed to share() is not reported shared. */
        TEST_F(Medium_ClusterBinding, AnUnsharedPackIsNotShared)
        {
            ClusterBinding binding = make_binding();
            const TensorPack shared(*runtime_, sixteen_floats(), 0);
            const TensorPack unshared(*runtime_, sixteen_floats(), 0);

            binding.share(shared);

            EXPECT_FALSE(binding.is_shared(unshared));
        }

        /** @brief Tests that no worker is known for an operation before any worker joins. */
        TEST_F(Medium_ClusterBinding, HasNoWorkerBeforeOneJoins)
        {
            const ClusterBinding binding = make_binding();

            EXPECT_FALSE(binding.has_worker_for(SushiRuntime::OperationID(1)));
        }

        /** @brief Tests that a null kernel registry is refused, with no runtime or cluster. */
        TEST(Small_ClusterBinding, RefusesANullRegistry)
        {
            EXPECT_THROW(ClusterBinding::require_kernels(SushiRuntime::sushi_ptr<KernelRegistry>()),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
