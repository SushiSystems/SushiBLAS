/**************************************************************************/
/* test_bound_pack.cpp                                                    */
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
 * @file test_bound_pack.cpp
 * @brief Tests that bind_views() checks views against layouts and hands segments back typed.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/remote_task.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/distributed/buffer_view.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using SushiRuntime::Distributed::BufferView;

            /** @brief Returns the layout of a float tensor of 3 and an int64 tensor of 2. */
            PackLayout small_layout()
            {
                const std::int64_t three[] = {3};
                const std::int64_t two[] = {2};
                const std::vector<Tensor> tensors = {
                    Tensor(SushiRuntime::sushi_ptr<Storage>(),
                           SushiRuntime::span<const std::int64_t>(three, 1),
                           Core::DataType::Float32),
                    Tensor(SushiRuntime::sushi_ptr<Storage>(),
                           SushiRuntime::span<const std::int64_t>(two, 1),
                           Core::DataType::Int64)};
                return PackLayout::of(
                    SushiRuntime::span<const Tensor>(tensors.data(), tensors.size()));
            }
        } // namespace

        /**
         * @brief Tests that a bound pack lays out a float tensor of 3 and an int64 tensor of 2 in
         *        two 128-byte aligned segments, 256 bytes in all.
         */
        TEST(Small_BoundPack, SmallLayoutHoldsTwoAlignedSegments)
        {
            const PackLayout layout = small_layout();

            EXPECT_EQ(layout.size_bytes(), 256U);
        }

        /** @brief Tests that the int64 segment starts at byte 128 and counts two elements. */
        TEST(Small_BoundPack, SegmentIsNamedAtItsLayoutOffset)
        {
            const PackLayout layout = small_layout();
            std::vector<std::byte> memory(256);
            const BufferView view{memory.data(), memory.size()};

            const std::vector<BoundPack> bound =
                bind_views(SushiRuntime::span<const BufferView>(&view, 1),
                           SushiRuntime::span<const PackLayout>(&layout, 1));
            ASSERT_EQ(bound.size(), 1U);

            const PackSpan longs = bound[0].segment(Core::DataType::Int64);
            EXPECT_EQ(longs.count, 2U);
            EXPECT_EQ(static_cast<void*>(longs.as<std::int64_t>()),
                      static_cast<void*>(memory.data() + 128));
        }

        /** @brief Tests that tensor(0) is the float tensor, at byte 0 and counting three. */
        TEST(Small_BoundPack, TensorLiesAtItsEntryOffset)
        {
            const PackLayout layout = small_layout();
            std::vector<std::byte> memory(256);
            const BufferView view{memory.data(), memory.size()};

            const std::vector<BoundPack> bound =
                bind_views(SushiRuntime::span<const BufferView>(&view, 1),
                           SushiRuntime::span<const PackLayout>(&layout, 1));
            ASSERT_EQ(bound.size(), 1U);

            const PackSpan floats = bound[0].tensor(0);
            EXPECT_EQ(floats.count, 3U);
            EXPECT_EQ(static_cast<void*>(floats.as<float>()), static_cast<void*>(memory.data()));
        }

        /** @brief Tests that tensor(1) is the int64 tensor, at byte 128 and counting two. */
        TEST(Small_BoundPack, SecondTensorLiesAtItsSegmentOffset)
        {
            const PackLayout layout = small_layout();
            std::vector<std::byte> memory(256);
            const BufferView view{memory.data(), memory.size()};

            const std::vector<BoundPack> bound =
                bind_views(SushiRuntime::span<const BufferView>(&view, 1),
                           SushiRuntime::span<const PackLayout>(&layout, 1));
            ASSERT_EQ(bound.size(), 1U);

            const PackSpan longs = bound[0].tensor(1);
            EXPECT_EQ(longs.count, 2U);
            EXPECT_EQ(static_cast<void*>(longs.as<std::int64_t>()),
                      static_cast<void*>(memory.data() + 128));
        }

        /** @brief Tests that the segment of a type no tensor has is empty with a null pointer. */
        TEST(Small_BoundPack, AbsentTypeGivesANullSegment)
        {
            const PackLayout layout = small_layout();
            std::vector<std::byte> memory(256);
            const BufferView view{memory.data(), memory.size()};

            const std::vector<BoundPack> bound =
                bind_views(SushiRuntime::span<const BufferView>(&view, 1),
                           SushiRuntime::span<const PackLayout>(&layout, 1));
            ASSERT_EQ(bound.size(), 1U);

            EXPECT_EQ(bound[0].segment(Core::DataType::Half).data, nullptr);
        }

        /** @brief Tests that reading a segment as another element type than its own throws. */
        TEST(Small_BoundPack, SegmentRefusesAnotherElementType)
        {
            const PackLayout layout = small_layout();
            std::vector<std::byte> memory(256);
            const BufferView view{memory.data(), memory.size()};

            const std::vector<BoundPack> bound =
                bind_views(SushiRuntime::span<const BufferView>(&view, 1),
                           SushiRuntime::span<const PackLayout>(&layout, 1));
            ASSERT_EQ(bound.size(), 1U);

            const PackSpan longs = bound[0].segment(Core::DataType::Int64);
            EXPECT_THROW((void)longs.as<float>(), std::runtime_error);
        }

        /** @brief Tests that bind_views() refuses a view whose size differs from its layout's. */
        TEST(Small_BoundPack, BindViewsRefusesAViewOfAnotherSize)
        {
            const PackLayout layout = small_layout();
            std::vector<std::byte> memory(layout.size_bytes() + 128);
            const BufferView view{memory.data(), memory.size()};

            EXPECT_THROW((void)bind_views(SushiRuntime::span<const BufferView>(&view, 1),
                                          SushiRuntime::span<const PackLayout>(&layout, 1)),
                         std::runtime_error);
        }

        /** @brief Tests that bind_views() refuses a count of views other than the layouts'. */
        TEST(Small_BoundPack, BindViewsRefusesAnotherViewCount)
        {
            const PackLayout layout = small_layout();
            EXPECT_THROW((void)bind_views(SushiRuntime::span<const BufferView>(),
                                          SushiRuntime::span<const PackLayout>(&layout, 1)),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
