/**************************************************************************/
/* test_tensor_factory_threads.cpp                                        */
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
 * @file test_tensor_factory_threads.cpp
 * @brief Tests that tensors are created and released from several threads at once.
 * @author Mustafa Garip
 */

#include <vector>
#include <future>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds how many threads create tensors at the same time.
            constexpr std::size_t K_THREADS = 8;

            /// @brief Holds how many tensors each thread creates.
            constexpr std::size_t K_TENSORS_PER_THREAD = 256;

            /// @brief Holds the number of sizes the cycle visits: 2^0 to 2^18 FLOAT32 elements.
            constexpr std::size_t K_SIZE_STEPS = 19;

            /// @brief Holds the byte interval one live storage occupies.
            struct ByteRange
            {
                std::uintptr_t begin; /**< Address of the first byte. */
                std::uintptr_t end;   /**< Address one past the last byte. */
            };

            /// @brief Creates count FLOAT32 tensors of 1 element up to 1 MiB, cycling through the sizes.
            std::vector<Tensor> create_tensors(const TensorFactory& factory, std::size_t count)
            {
                std::vector<Tensor> tensors;
                tensors.reserve(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::vector<int64_t> dims{int64_t{1} << (i % K_SIZE_STEPS)};
                    tensors.push_back(factory.create(SushiRuntime::span<const int64_t>(dims.data(),
                                                                                       dims.size()),
                                                     Core::DataType::Float32));
                }
                return tensors;
            }

            /// @brief Starts one thread per slot, each creating K_TENSORS_PER_THREAD tensors.
            std::vector<std::future<std::vector<Tensor>>> start_creators(
                const TensorFactory& factory)
            {
                std::vector<std::future<std::vector<Tensor>>> creators;
                for (std::size_t thread = 0; thread < K_THREADS; ++thread)
                {
                    creators.push_back(std::async(std::launch::async, [&factory]
                    {
                        return create_tensors(factory, K_TENSORS_PER_THREAD);
                    }));
                }
                return creators;
            }

            /// @brief Waits for every creator and returns the tensors of all of them.
            std::vector<Tensor> collect(std::vector<std::future<std::vector<Tensor>>>& creators)
            {
                std::vector<Tensor> all;
                for (std::future<std::vector<Tensor>>& creator : creators)
                {
                    std::vector<Tensor> made = creator.get();
                    all.insert(all.end(), made.begin(), made.end());
                }
                return all;
            }

            /// @brief Returns how many tensors start at an address K_STORAGE_ALIGNMENT does not divide.
            std::size_t count_misaligned(const std::vector<Tensor>& tensors)
            {
                std::size_t misaligned = 0;
                for (const Tensor& tensor : tensors)
                {
                    if (reinterpret_cast<std::uintptr_t>(tensor.data()) % K_STORAGE_ALIGNMENT != 0)
                        ++misaligned;
                }
                return misaligned;
            }

            /// @brief Returns the byte interval of every tensor's storage, ordered by address.
            std::vector<ByteRange> sorted_ranges(const std::vector<Tensor>& tensors)
            {
                std::vector<ByteRange> ranges;
                ranges.reserve(tensors.size());
                for (const Tensor& tensor : tensors)
                {
                    const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data());
                    ranges.push_back({begin, begin + tensor.storage->size_bytes()});
                }
                std::sort(ranges.begin(), ranges.end(), [](const ByteRange& a, const ByteRange& b)
                          { return a.begin < b.begin; });
                return ranges;
            }

            /// @brief Returns how many neighbours in an ordered list of ranges share a byte.
            std::size_t count_overlaps(const std::vector<ByteRange>& ordered)
            {
                std::size_t overlaps = 0;
                for (std::size_t i = 1; i < ordered.size(); ++i)
                {
                    if (ordered[i].begin < ordered[i - 1].end)
                        ++overlaps;
                }
                return overlaps;
            }
        } // namespace

        /** @brief Tests that eight threads create 256 tensors each, from 1 element to 1 MiB, and that none share a byte. */
        TEST(Medium_TensorFactoryThreads, ConcurrentCreatesAreAlignedDisjointAndReleased)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const std::size_t before = TestEnvironment::live_allocation_count();

            {
                std::vector<std::future<std::vector<Tensor>>> creators = start_creators(factory);
                const std::vector<Tensor> tensors = collect(creators);

                EXPECT_EQ(tensors.size(), K_THREADS * K_TENSORS_PER_THREAD);
                EXPECT_EQ(count_misaligned(tensors), std::size_t{0});
                EXPECT_EQ(count_overlaps(sorted_ranges(tensors)), std::size_t{0});
                EXPECT_EQ(TestEnvironment::live_allocation_count(),
                          before + K_THREADS * K_TENSORS_PER_THREAD);
            }

            EXPECT_EQ(TestEnvironment::live_allocation_count(), before);
        }
    } // namespace Test
} // namespace SushiBLAS
