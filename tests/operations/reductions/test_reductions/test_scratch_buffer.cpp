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
 * @file test_scratch_buffer.cpp
 * @brief Tests ScratchBuffer, the lease on one pooled device scratch block.
 * @author Mustafa Garip
 */

#include <array>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
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

            /// @brief Holds how many leases are alive at once in LiveLeasesNeverShareABlock.
            constexpr std::size_t K_LIVE_LEASES = 300;

            /// @brief Holds how many threads lease at once in LeasesAreSafeAcrossThreads.
            constexpr std::size_t K_THREADS = 4;

            /// @brief Holds how many buffers each thread leases and releases.
            constexpr std::size_t K_LEASES_PER_THREAD = 1000;

            /// @brief Holds the byte a write over a lease is read back as.
            constexpr unsigned char K_PATTERN = 0x5A;

            /**
             * @brief Leases and copies buffers from pool on K_THREADS threads and counts the empty ones.
             *
             * GoogleTest assertions are not thread-safe on Windows, so a worker only counts.
             * @return How many leases held no block.
             */
            std::size_t count_empty_leases_across_threads(ScratchPool& pool)
            {
                const std::array<std::size_t, 6> requests = {128, 256, 512, 1024, 2048, 4096};
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

                return empty_leases.load();
            }

            /** @brief Takes count leases of 4096 bytes at once and returns the addresses they hold. */
            std::unordered_set<const void*> addresses_of_live_leases(ScratchPool& pool,
                                                                     std::size_t count)
            {
                std::vector<ScratchBuffer> leases;
                leases.reserve(count);
                std::unordered_set<const void*> addresses;
                addresses.reserve(count);

                for (std::size_t i = 0; i < count; ++i)
                {
                    leases.emplace_back(pool, std::size_t{4096}, K_DEVICE);
                    addresses.insert(leases.back().data());
                }
                return addresses;
            }
        } // namespace

        /** @brief Gives each test the fixture's engine and the test environment's runtime. */
        class Medium_ScratchBuffer : public SushiBLASTest
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
        };

        /** @brief Records reductions of a 2^20 input, and skips the case on a device that plans no partials for it. */
        class Medium_ScratchBufferReduction : public Medium_ScratchBuffer
        {
            protected:
                /** @brief Builds the input and the result, or skips the test when no partials are planned. */
                void SetUp() override
                {
                    Medium_ScratchBuffer::SetUp();

                    input_ = make_reduction_input();
                    result_ = engine->create_tensor({1});
                    if (Dispatch::planned_partials_for(*input_) == 0)
                        GTEST_SKIP() << "This device plans no partials for 2^20 elements.";
                }

                /// @brief The 2^20 ones the reductions sum.
                std::optional<Tensor> input_;

                /// @brief The scalar the reductions write.
                std::optional<Tensor> result_;
        };

        TEST_F(Medium_ScratchBuffer, ZeroBytesLeaseNothing)
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

        /** @brief One request size and the size class of the block that serves it. */
        struct ClassRow
        {
            std::string name;
            std::size_t request;
            std::size_t block_bytes;
        };

        /** @brief Returns the requests of 1, 129 and 4096 bytes with the blocks that serve them. */
        static std::vector<ClassRow> class_rows()
        {
            return {
                {"OneByte", 1, 128},
                {"OneByteOverTheFirstClass", 129, 256},
                {"AWholeClass", 4096, 4096},
            };
        }

        /** @brief Leases one request size from a fresh pool. */
        class Medium_ScratchBufferClass : public Medium_ScratchBuffer,
                                          public ::testing::WithParamInterface<ClassRow>
        {
        };

        TEST_P(Medium_ScratchBufferClass, ALeaseHoldsABlockOfItsClass)
        {
            const ClassRow& row = GetParam();
            ScratchPool pool(TestEnvironment::runtime());

            const ScratchBuffer lease(pool, row.request, K_DEVICE);

            EXPECT_FALSE(lease.is_empty());
            EXPECT_NE(lease.data(), nullptr);
            EXPECT_EQ(lease.bytes(), row.request);
            ASSERT_NE(lease.storage().get(), nullptr);
            EXPECT_EQ(lease.storage()->size_bytes(), row.block_bytes);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_ScratchBufferClass, ::testing::ValuesIn(class_rows()),
                                 [](const ::testing::TestParamInfo<ClassRow>& info)
                                 {
                                     return info.param.name;
                                 });

        TEST_F(Medium_ScratchBuffer, ACopyHoldsTheBlockOfItsOriginal)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const ScratchBuffer original(pool, std::size_t{128}, K_DEVICE);

            const ScratchBuffer copy(original);

            EXPECT_EQ(copy.data(), original.data());
        }

        TEST_F(Medium_ScratchBuffer, ACopyKeepsTheBlockOutOfThePoolWhenTheOriginalIsReleased)
        {
            ScratchPool pool(TestEnvironment::runtime());

            std::optional<ScratchBuffer> original(std::in_place, pool, std::size_t{128}, K_DEVICE);
            const ScratchBuffer copy(*original);
            original.reset();

            const ScratchBuffer second(pool, 128, K_DEVICE);

            EXPECT_NE(second.data(), copy.data());
            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
        }

        TEST_F(Medium_ScratchBuffer, NoBlockReturnsToThePoolWhileACopyOrAnAssignmentHoldsIt)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t idle_baseline = pool.idle_block_count();

            const ScratchBuffer original(pool, 512, K_DEVICE);
            const ScratchBuffer copy(original);
            ScratchBuffer assigned;
            assigned = copy;

            EXPECT_EQ(pool.idle_block_count(), idle_baseline);
        }

        TEST_F(Medium_ScratchBuffer, TheLastCopyReturnsTheBlockToThePool)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t idle_baseline = pool.idle_block_count();

            {
                const ScratchBuffer original(pool, 512, K_DEVICE);
                const ScratchBuffer copy(original);
                ScratchBuffer assigned;
                assigned = copy;
            }

            EXPECT_EQ(pool.idle_block_count(), idle_baseline + 1);
        }

        TEST_F(Medium_ScratchBuffer, ABlockReturnedToThePoolIsHandedOutAgain)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const void* leased = nullptr;

            {
                const ScratchBuffer first(pool, 512, K_DEVICE);
                leased = first.data();
            }

            const ScratchBuffer next(pool, 512, K_DEVICE);
            EXPECT_EQ(next.data(), leased);
        }

        TEST_F(Medium_ScratchBuffer, ALeaseOutlivesItsPool)
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
            }

            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_baseline);
        }

        TEST_F(Medium_ScratchBuffer, ADeviceWriteOverALeaseSurvivesItsPool)
        {
            std::optional<ScratchPool> pool(std::in_place, TestEnvironment::runtime());
            const ScratchBuffer lease(*pool, 256, K_DEVICE);
            pool.reset();

            // The block is still allocated, so the device accepts a write over all of it.
            TestEnvironment::queue(K_DEVICE).memset(lease.data(), K_PATTERN, lease.bytes()).wait();
            std::vector<unsigned char> back(lease.bytes());
            TestEnvironment::queue(K_DEVICE).memcpy(back.data(), lease.data(), back.size()).wait();

            EXPECT_EQ(back, std::vector<unsigned char>(lease.bytes(), K_PATTERN));
        }

        TEST_F(Medium_ScratchBuffer, LiveLeasesNeverShareABlock)
        {
            ScratchPool pool(TestEnvironment::runtime());

            const std::unordered_set<const void*> addresses =
                addresses_of_live_leases(pool, K_LIVE_LEASES);

            EXPECT_EQ(addresses.size(), K_LIVE_LEASES);
            EXPECT_EQ(addresses.count(nullptr), std::size_t{0});
        }

        TEST_F(Medium_ScratchBuffer, LeasesAreSafeAcrossThreads)
        {
            ScratchPool pool(TestEnvironment::runtime());
            const std::size_t baseline =
                TestEnvironment::live_allocation_count() - pool.idle_block_count();

            const std::size_t empty_leases = count_empty_leases_across_threads(pool);

            EXPECT_EQ(empty_leases, std::size_t{0});
            EXPECT_EQ(TestEnvironment::live_allocation_count() - pool.idle_block_count(), baseline);
        }

        TEST_F(Medium_ScratchBufferReduction, ARecordedReductionLeasesOneBlock)
        {
            const std::size_t live_before_record = TestEnvironment::live_allocation_count();

            ReductionOps(recorder()).sum(*input_, *result_);

            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before_record + 1);
        }

        TEST_F(Medium_ScratchBufferReduction, ARunReductionKeepsItsBlockInThePool)
        {
            const std::size_t live_before_record = TestEnvironment::live_allocation_count();

            ReductionOps(recorder()).sum(*input_, *result_);
            recording->run().value();

            expect_tensor_equal(*result_, {static_cast<float>(K_REDUCTION_ELEMENTS)});
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before_record + 1);
        }

        TEST_F(Medium_ScratchBufferReduction, ASecondReductionReusesTheBlockOfTheFirst)
        {
            ReductionOps(recorder()).sum(*input_, *result_);
            recording->run().value();
            const std::size_t live_after_first = TestEnvironment::live_allocation_count();

            ReductionOps(recorder()).sum(*input_, *result_);
            recording->run().value();

            expect_tensor_equal(*result_, {static_cast<float>(K_REDUCTION_ELEMENTS)});
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_after_first);
        }

        TEST_F(Medium_ScratchBufferReduction, AClearedRecorderReturnsItsLeaseToThePool)
        {
            ScratchPool pool(TestEnvironment::runtime());
            SushiRuntime::Graph first_graph = TestEnvironment::runtime().graph();
            LateCountHold late_counts(TestEnvironment::runtime());
            TaskRecorder first_recorder(first_graph.recorder(), pool, late_counts);
            ReductionOps(first_recorder).sum(*input_, *result_);

            // Clearing the recorder destroys the recorded task, the only holder of the lease.
            first_graph.recorder().clear();

            EXPECT_EQ(pool.idle_block_count(), std::size_t{1});
        }

        TEST_F(Medium_ScratchBufferReduction, ASecondRecorderLeasesTheBlockTheFirstReturned)
        {
            Tensor second_result = engine->create_tensor({1});
            ScratchPool pool(TestEnvironment::runtime());
            SushiRuntime::Graph first_graph = TestEnvironment::runtime().graph();
            SushiRuntime::Graph second_graph = TestEnvironment::runtime().graph();
            LateCountHold late_counts(TestEnvironment::runtime());
            TaskRecorder first_recorder(first_graph.recorder(), pool, late_counts);
            TaskRecorder second_recorder(second_graph.recorder(), pool, late_counts);
            ReductionOps(first_recorder).sum(*input_, *result_);
            first_graph.recorder().clear();
            const std::size_t live_after_drop = TestEnvironment::live_allocation_count();

            ReductionOps(second_recorder).sum(*input_, second_result);

            EXPECT_EQ(pool.idle_block_count(), std::size_t{0});
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_after_drop);
        }
    } // namespace Test
} // namespace SushiBLAS
