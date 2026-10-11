/**************************************************************************/
/* test_scratch_buffer.cpp                                                */
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
 * @file test_scratch_buffer.cpp
 * @brief Tests ScratchBuffer, the lease on one pooled device scratch block.
 * @author Mustafa Garip
 */

#include <array>
#include <atomic>
#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include "../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the device queue every direct lease of this file is taken on.
            constexpr std::size_t K_DEVICE = 0;

            /// @brief Holds a reduction input's element count: 2^20, above the CPU two-stage floor.
            constexpr std::int64_t K_REDUCTION_ELEMENTS = std::int64_t{1} << 20;

            /// @brief Holds the absolute tolerance of a sum of 2^20 ones, exact in FLOAT32.
            constexpr double K_SUM_TOLERANCE = 0.0;

            /// @brief Holds how many leases are alive at once in LiveLeasesNeverShareABlock.
            constexpr std::size_t K_LIVE_LEASES = 300;

            /// @brief Holds how many threads lease at once in LeasesAreSafeAcrossThreads.
            constexpr std::size_t K_THREADS = 4;

            /// @brief Holds how many buffers each thread leases and releases.
            constexpr std::size_t K_LEASES_PER_THREAD = 1000;
        } // namespace

        /** @brief Gives each test the fixture's engine and the test environment's runtime. */
        class Unit_ScratchBuffer : public SushiBLASTest
        {
            protected:
                /** @brief Returns a FLOAT32 tensor of K_REDUCTION_ELEMENTS ones. */
                [[nodiscard]] Tensor make_reduction_input()
                {
                    Tensor input = engine->create_tensor({K_REDUCTION_ELEMENTS});
                    fill_tensor(input, std::vector<float>(
                                           static_cast<std::size_t>(K_REDUCTION_ELEMENTS), 1.0f));
                    return input;
                }

                /** @brief Returns whether a whole-tensor reduction of input leases scratch. */
                [[nodiscard]] static bool plans_partials(const Tensor& input)
                {
                    return Dispatch::planned_partials_for(input) > 0;
                }
        };

        TEST_F(Unit_ScratchBuffer, ZeroBytesLeaseNothing)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t live_baseline = TestEnvironment::live_allocation_count();
            const std::size_t idle_baseline = pool.idle_block_count();

            {
                const ScratchBuffer lease(pool, 0, K_DEVICE);

                EXPECT_TRUE(lease.is_empty());
                EXPECT_EQ(lease.data(), nullptr);
                EXPECT_EQ(lease.bytes(), std::size_t{0});
                EXPECT_EQ(lease.storage().get(), nullptr);
                EXPECT_EQ(TestEnvironment::live_allocation_count(), live_baseline);
            }

            EXPECT_EQ(pool.idle_block_count(), idle_baseline);
        }

        TEST_F(Unit_ScratchBuffer, ALeaseHoldsABlockOfItsClass)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::array<std::size_t, 3> requests = {1, 129, 4096};
            const std::array<std::size_t, 3> block_bytes = {128, 256, 4096};

            for (std::size_t i = 0; i < requests.size(); ++i)
            {
                const ScratchBuffer lease(pool, requests[i], K_DEVICE);

                EXPECT_FALSE(lease.is_empty());
                EXPECT_NE(lease.data(), nullptr);
                EXPECT_EQ(lease.bytes(), requests[i]);
                ASSERT_NE(lease.storage().get(), nullptr);
                EXPECT_EQ(lease.storage()->size_bytes(), block_bytes[i])
                    << "request of " << requests[i] << " bytes";
            }
        }

        TEST_F(Unit_ScratchBuffer, ACopyKeepsTheBlockOut)
        {
            ScratchPool pool(TestEnvironment::runtime());

            std::optional<ScratchBuffer> original(std::in_place, pool, std::size_t{128}, K_DEVICE);
            const ScratchBuffer copy(*original);
            ASSERT_EQ(copy.data(), original->data());
            original.reset();

            const ScratchBuffer second(pool, 128, K_DEVICE);

            EXPECT_NE(second.data(), copy.data());
            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
        }

        TEST_F(Unit_ScratchBuffer, TheLastCopyReturnsTheBlock)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t idle_baseline = pool.idle_block_count();
            const void* leased = nullptr;

            {
                const ScratchBuffer original(pool, 512, K_DEVICE);
                const ScratchBuffer copy(original);
                ScratchBuffer assigned;
                assigned = copy;
                leased = original.data();

                EXPECT_EQ(pool.idle_block_count(), idle_baseline);
            }

            EXPECT_EQ(pool.idle_block_count(), idle_baseline + 1);

            const ScratchBuffer next(pool, 512, K_DEVICE);
            EXPECT_EQ(next.data(), leased);
            EXPECT_EQ(pool.idle_block_count(), idle_baseline);
        }

        TEST_F(Unit_ScratchBuffer, ALeaseOutlivesItsPool)
        {
            const std::size_t live_baseline = TestEnvironment::live_allocation_count();

            {
                std::optional<ScratchPool> pool(std::in_place, TestEnvironment::runtime());
                const ScratchBuffer lease(*pool, 256, K_DEVICE);
                const void* leased = lease.data();
                pool.reset();

                EXPECT_EQ(lease.data(), leased);
                EXPECT_EQ(lease.bytes(), std::size_t{256});
                EXPECT_EQ(TestEnvironment::live_allocation_count(), live_baseline + 1);

                // The block is still allocated, so the device accepts a write over all of it.
                TestEnvironment::queue(K_DEVICE).memset(lease.data(), 0, lease.bytes()).wait();
            }

            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_baseline);
        }

        TEST_F(Unit_ScratchBuffer, LiveLeasesNeverShareABlock)
        {
            ScratchPool pool(TestEnvironment::runtime());

            std::vector<ScratchBuffer> leases;
            leases.reserve(K_LIVE_LEASES);
            std::unordered_set<const void*> addresses;
            addresses.reserve(K_LIVE_LEASES);

            for (std::size_t i = 0; i < K_LIVE_LEASES; ++i)
            {
                leases.emplace_back(pool, std::size_t{4096}, K_DEVICE);
                addresses.insert(leases.back().data());
            }

            EXPECT_EQ(addresses.size(), K_LIVE_LEASES);
            EXPECT_EQ(addresses.count(nullptr), std::size_t{0});
        }

        TEST_F(Unit_ScratchBuffer, LeasesAreSafeAcrossThreads)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t baseline = TestEnvironment::live_allocation_count() - pool.idle_block_count();
            const std::array<std::size_t, 6> requests = {128, 256, 512, 1024, 2048, 4096};

            // GoogleTest assertions are not thread-safe on Windows, so a worker only counts.
            std::atomic<std::size_t> empty_leases{0};

            std::vector<std::thread> workers;
            workers.reserve(K_THREADS);
            for (std::size_t worker = 0; worker < K_THREADS; ++worker)
            {
                workers.emplace_back([&pool, &requests, &empty_leases, worker]()
                {
                    for (std::size_t i = 0; i < K_LEASES_PER_THREAD; ++i)
                    {
                        const ScratchBuffer lease(pool, requests[(i + worker) % requests.size()],
                                                  K_DEVICE);
                        const ScratchBuffer copy(lease);
                        if (copy.data() == nullptr)
                            empty_leases.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            }
            for (std::thread& worker : workers)
                worker.join();

            EXPECT_EQ(empty_leases.load(), std::size_t{0});
            EXPECT_EQ(TestEnvironment::live_allocation_count() - pool.idle_block_count(), baseline);
        }

        TEST_F(Unit_ScratchBuffer, ARecordedReductionReusesItsScratch)
        {
            const Tensor input = make_reduction_input();
            Tensor result = engine->create_tensor({1});
            if (!plans_partials(input))
                GTEST_SKIP() << "This device plans no partials for 2^20 elements.";

            const std::size_t live_before_record = TestEnvironment::live_allocation_count();
            ReductionOps(*engine).sum(input, result);
            ASSERT_EQ(TestEnvironment::live_allocation_count(), live_before_record + 1);

            engine->execute().value();
            const std::size_t live_after_first = TestEnvironment::live_allocation_count();
            verify_tensor(result, {static_cast<float>(K_REDUCTION_ELEMENTS)}, K_SUM_TOLERANCE);

            ReductionOps(*engine).sum(input, result);
            engine->execute().value();
            const std::size_t live_after_second = TestEnvironment::live_allocation_count();
            verify_tensor(result, {static_cast<float>(K_REDUCTION_ELEMENTS)}, K_SUM_TOLERANCE);

            EXPECT_EQ(live_after_first, live_before_record + 1);
            EXPECT_EQ(live_after_second, live_after_first);
        }

        TEST_F(Unit_ScratchBuffer, TwoRecordersLeaseFromOnePool)
        {
            const Tensor input = make_reduction_input();
            Tensor first_result = engine->create_tensor({1});
            Tensor second_result = engine->create_tensor({1});
            if (!plans_partials(input))
                GTEST_SKIP() << "This device plans no partials for 2^20 elements.";

            ScratchPool pool(TestEnvironment::runtime());
            SushiRuntime::Graph first_graph = TestEnvironment::runtime().graph();
            SushiRuntime::Graph second_graph = TestEnvironment::runtime().graph();
            TaskRecorder first_recorder(first_graph.recorder(), pool);
            TaskRecorder second_recorder(second_graph.recorder(), pool);

            const std::size_t live_before_record = TestEnvironment::live_allocation_count();
            ReductionOps(first_recorder).sum(input, first_result);
            ASSERT_EQ(TestEnvironment::live_allocation_count(), live_before_record + 1);

            // Clearing the recorder destroys the recorded task, the only holder of the lease.
            first_graph.recorder().clear();
            ASSERT_EQ(pool.idle_block_count(), std::size_t{1});
            const std::size_t live_after_drop = TestEnvironment::live_allocation_count();

            ReductionOps(second_recorder).sum(input, second_result);

            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_after_drop);
        }
    } // namespace Test
} // namespace SushiBLAS
