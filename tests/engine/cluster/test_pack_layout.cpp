/**************************************************************************/
/* test_pack_layout.cpp                                                   */
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
 * @file test_pack_layout.cpp
 * @brief Tests that PackLayout groups tensors by type, aligns segments and fingerprints the layout.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a tensor with no storage, which a layout reads only the shape of. */
            Tensor shaped(std::vector<std::int64_t> dims, Core::DataType dtype,
                          Core::Layout layout = Core::Layout::RowMajor)
            {
                return Tensor(SushiRuntime::sushi_ptr<Storage>(),
                              SushiRuntime::span<const std::int64_t>(dims.data(), dims.size()),
                              dtype, 0, layout);
            }

            /** @brief Returns the layout of a list of tensors. */
            PackLayout layout_of(const std::vector<Tensor>& tensors)
            {
                return PackLayout::of(
                    SushiRuntime::span<const Tensor>(tensors.data(), tensors.size()));
            }
        } // namespace

        /** @brief Tests that tensors of one type share a segment in the order given. */
        TEST(Small_PackLayout, GroupsTensorsOfOneTypeInGivenOrder)
        {
            const std::vector<Tensor> tensors = {shaped({3}, Core::DataType::Float32),
                                                 shaped({2}, Core::DataType::Int64),
                                                 shaped({5}, Core::DataType::Float32)};
            const PackLayout layout = layout_of(tensors);

            const PackSegment& floats = layout.segment(Core::DataType::Float32);
            EXPECT_EQ(floats.size_bytes, 32U);
            EXPECT_EQ(floats.element_count, 8U);
            EXPECT_EQ(layout.entries()[0].offset_bytes, floats.offset_bytes);
            EXPECT_EQ(layout.entries()[2].offset_bytes, floats.offset_bytes + 12U);
            EXPECT_EQ(layout.entries()[1].offset_bytes,
                      layout.segment(Core::DataType::Int64).offset_bytes);
        }

        /** @brief Tests that segments follow the type order and start on 128-byte boundaries. */
        TEST(Small_PackLayout, AlignsSegmentsInTypeOrder)
        {
            const std::vector<Tensor> tensors = {shaped({1}, Core::DataType::Int32),
                                                 shaped({3}, Core::DataType::Half),
                                                 shaped({2}, Core::DataType::Complex64)};
            const PackLayout layout = layout_of(tensors);

            EXPECT_EQ(layout.segment(Core::DataType::Half).offset_bytes, 0U);
            EXPECT_EQ(layout.segment(Core::DataType::Complex64).offset_bytes, 128U);
            EXPECT_EQ(layout.segment(Core::DataType::Int32).offset_bytes, 256U);
            EXPECT_EQ(layout.size_bytes(), 384U);
            EXPECT_EQ(layout.segment(Core::DataType::Float64).size_bytes, 0U);
        }

        /** @brief Tests that UINT32, the last type, has a segment of its own after INT64. */
        TEST(Small_PackLayout, GivesUInt32ASegmentAfterInt64)
        {
            const std::vector<Tensor> tensors = {shaped({3}, Core::DataType::UInt32),
                                                 shaped({2}, Core::DataType::Int64)};
            const PackLayout layout = layout_of(tensors);

            EXPECT_EQ(layout.segment(Core::DataType::Int64).offset_bytes, 0U);
            EXPECT_EQ(layout.segment(Core::DataType::UInt32).offset_bytes, 128U);
            EXPECT_EQ(layout.segment(Core::DataType::UInt32).size_bytes, 12U);
            EXPECT_NE(layout.fingerprint(),
                      layout_of({shaped({3}, Core::DataType::Int32),
                                 shaped({2}, Core::DataType::Int64)}).fingerprint());
        }

        /** @brief Tests that the same tensors give the same fingerprint. */
        TEST(Small_PackLayout, FingerprintIsStableForEqualLayouts)
        {
            const std::vector<Tensor> tensors = {shaped({4, 2}, Core::DataType::Float64),
                                                 shaped({7}, Core::DataType::Int32)};
            EXPECT_EQ(layout_of(tensors).fingerprint(), layout_of(tensors).fingerprint());
        }

        /** @brief Tests that a change of shape, type, layout or order changes the fingerprint. */
        TEST(Small_PackLayout, FingerprintChangesWithShapeTypeLayoutAndOrder)
        {
            const std::uint64_t base =
                layout_of({shaped({4, 2}, Core::DataType::Float32),
                           shaped({7}, Core::DataType::Float32)}).fingerprint();

            EXPECT_NE(base, layout_of({shaped({2, 4}, Core::DataType::Float32),
                                       shaped({7}, Core::DataType::Float32)}).fingerprint());
            EXPECT_NE(base, layout_of({shaped({4, 2}, Core::DataType::Int32),
                                       shaped({7}, Core::DataType::Float32)}).fingerprint());
            EXPECT_NE(base,
                      layout_of({shaped({4, 2}, Core::DataType::Float32, Core::Layout::ColumnMajor),
                                 shaped({7}, Core::DataType::Float32)}).fingerprint());
            EXPECT_NE(base, layout_of({shaped({7}, Core::DataType::Float32),
                                       shaped({4, 2}, Core::DataType::Float32)}).fingerprint());
        }

        /** @brief Tests that a layout refuses an empty list and a list with no element. */
        TEST(Small_PackLayout, RefusesAPackWithNoElement)
        {
            EXPECT_THROW((void)layout_of({}), std::runtime_error);
            EXPECT_THROW((void)layout_of({shaped({0}, Core::DataType::Float32)}),
                         std::runtime_error);
        }

        /** @brief Tests that matches() compares type, layout and shape. */
        TEST(Small_PackLayout, MatchesComparesTypeLayoutAndShape)
        {
            const PackLayout layout = layout_of({shaped({3, 2}, Core::DataType::Float32)});
            const PackEntry& entry = layout.entries()[0];

            EXPECT_TRUE(PackLayout::matches(entry, shaped({3, 2}, Core::DataType::Float32)));
            EXPECT_FALSE(PackLayout::matches(entry, shaped({2, 3}, Core::DataType::Float32)));
            EXPECT_FALSE(PackLayout::matches(entry, shaped({3, 2}, Core::DataType::Float64)));
            EXPECT_FALSE(PackLayout::matches(
                entry, shaped({3, 2}, Core::DataType::Float32, Core::Layout::ColumnMajor)));
        }
    } // namespace Test
} // namespace SushiBLAS
