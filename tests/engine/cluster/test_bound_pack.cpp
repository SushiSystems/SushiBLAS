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

        /** @brief Tests that a bound pack names each segment and tensor at its offset. */
        TEST(Small_BoundPack, NamesSegmentsAndTensorsTyped)
        {
            const PackLayout layout = small_layout();
            std::vector<std::byte> memory(layout.size_bytes());
            const BufferView view{memory.data(), memory.size()};

            const std::vector<BoundPack> bound =
                bind_views(SushiRuntime::span<const BufferView>(&view, 1),
                           SushiRuntime::span<const PackLayout>(&layout, 1));
            ASSERT_EQ(bound.size(), 1U);

            const PackSpan longs = bound[0].segment(Core::DataType::Int64);
            EXPECT_EQ(longs.count, 2U);
            EXPECT_EQ(static_cast<void*>(longs.as<std::int64_t>()),
                      static_cast<void*>(memory.data() +
                                         layout.segment(Core::DataType::Int64).offset_bytes));
            EXPECT_EQ(bound[0].tensor(0).count, 3U);
            EXPECT_EQ(bound[0].segment(Core::DataType::Half).data, nullptr);
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
