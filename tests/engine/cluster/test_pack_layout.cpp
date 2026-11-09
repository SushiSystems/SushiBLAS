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
#include <SushiBLAS/tensor/host_block.hpp>
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

            /** @brief Returns a contiguous float tensor of count elements on host storage. */
            Tensor host_floats(std::int64_t count)
            {
                const std::int64_t dims[] = {count};
                return Tensor(make_host_storage(static_cast<std::size_t>(count) * sizeof(float)),
                              SushiRuntime::span<const std::int64_t>(dims, 1),
                              Core::DataType::Float32);
            }

            /** @brief Runs require_tensors on a list against the layout of one float tensor of four. */
            void require_against_four_floats(const std::vector<Tensor>& tensors,
                                             std::size_t device_index)
            {
                layout_of({host_floats(4)}).require_tensors(
                    SushiRuntime::span<const Tensor>(tensors.data(), tensors.size()), device_index,
                    "test.pack");
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

        /** @brief Tests that segments follow the type order and start on 128-byte multiples. */
        TEST(Small_PackLayout, PlacesSegmentsInTypeOrderOn128ByteMultiples)
        {
            const PackLayout layout = layout_of({shaped({1}, Core::DataType::Int32),
                                                 shaped({3}, Core::DataType::Half),
                                                 shaped({2}, Core::DataType::Complex64)});

            EXPECT_EQ(layout.segment(Core::DataType::Half).offset_bytes, 0U);
            EXPECT_EQ(layout.segment(Core::DataType::Complex64).offset_bytes, 128U);
            EXPECT_EQ(layout.segment(Core::DataType::Int32).offset_bytes, 256U);
        }

        /**
         * @brief Tests that the size ends at the aligned end of the last non-empty segment and a
         *        type with no tensor takes no bytes.
         */
        TEST(Small_PackLayout, SizeEndsAtTheAlignedEndOfTheLastSegment)
        {
            // The Int32 segment starts at 256 and holds 4 bytes; 260 rounds up to 384.
            const PackLayout layout = layout_of({shaped({1}, Core::DataType::Int32),
                                                 shaped({3}, Core::DataType::Half),
                                                 shaped({2}, Core::DataType::Complex64)});

            EXPECT_EQ(layout.size_bytes(), 384U);
            EXPECT_EQ(layout.segment(Core::DataType::Float64).size_bytes, 0U);
        }

        /** @brief Tests that UInt32, the last type, has a segment of its own after Int64. */
        TEST(Small_PackLayout, GivesUInt32ASegmentAfterInt64)
        {
            const PackLayout layout = layout_of({shaped({3}, Core::DataType::UInt32),
                                                 shaped({2}, Core::DataType::Int64)});

            EXPECT_EQ(layout.segment(Core::DataType::UInt32).offset_bytes, 128U);
            EXPECT_EQ(layout.segment(Core::DataType::UInt32).size_bytes, 12U);
        }

        /** @brief Tests that the fingerprint changes when one tensor changes its type. */
        TEST(Small_PackLayout, FingerprintChangesWhenATypeChanges)
        {
            const PackLayout unsigned_pack = layout_of({shaped({3}, Core::DataType::UInt32),
                                                        shaped({2}, Core::DataType::Int64)});
            const PackLayout signed_pack = layout_of({shaped({3}, Core::DataType::Int32),
                                                      shaped({2}, Core::DataType::Int64)});

            EXPECT_NE(unsigned_pack.fingerprint(), signed_pack.fingerprint());
        }

        /** @brief Tests that tensors built apart with the same shapes give the same fingerprint. */
        TEST(Small_PackLayout, FingerprintIsStableForEqualLayouts)
        {
            const PackLayout first = layout_of({shaped({4, 2}, Core::DataType::Float64),
                                                shaped({7}, Core::DataType::Int32)});
            const PackLayout second = layout_of({shaped({4, 2}, Core::DataType::Float64),
                                                 shaped({7}, Core::DataType::Int32)});

            EXPECT_EQ(first.fingerprint(), second.fingerprint());
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

        /** @brief Tests that a type outside the enumeration is refused, not wrapped onto Half. */
        TEST(Small_Regression_PackLayout, SegmentRefusesADataTypeOutsideTheEnumeration)
        {
            const PackLayout layout = layout_of({shaped({4}, Core::DataType::Half)});
            const auto outside = static_cast<Core::DataType>(K_PACK_SEGMENT_COUNT);
            EXPECT_THROW((void)layout.segment(outside), std::runtime_error);
            EXPECT_EQ(layout.segment(Core::DataType::Half).size_bytes, 4 * sizeof(std::uint16_t));
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

        /** @brief Tests that require_tensors refuses a tensor of another shape. */
        TEST(Small_PackLayout, RequireTensorsRefusesATensorOfAnotherShape)
        {
            EXPECT_THROW(require_against_four_floats({host_floats(5)}, 0), std::runtime_error);
        }

        /** @brief Tests that require_tensors refuses a list of another length. */
        TEST(Small_PackLayout, RequireTensorsRefusesAListOfAnotherLength)
        {
            EXPECT_THROW(require_against_four_floats({host_floats(4), host_floats(4)}, 0),
                         std::runtime_error);
        }

        /** @brief Tests that require_tensors refuses a tensor with no storage. */
        TEST(Small_PackLayout, RequireTensorsRefusesATensorWithNoStorage)
        {
            EXPECT_THROW(require_against_four_floats({shaped({4}, Core::DataType::Float32)}, 0),
                         std::runtime_error);
        }

        /** @brief Tests that require_tensors refuses a tensor on another device than the pack. */
        TEST(Small_Regression_PackLayout, RequireTensorsRefusesATensorOnAnotherDevice)
        {
            // Host storage lives on device 0, so a pack on device 1 differs from it.
            EXPECT_THROW(require_against_four_floats({host_floats(4)}, 1), std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
