/**************************************************************************/
/* test_tensor_view.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// Tensor views: does a view's metadata describe the memory it actually
// occupies?
//
// The question these tests ask of `slice` is not "are the values right" - a
// slice does not move anything, so there is nothing to get right in that sense.
// It is "do the strides the view reports lead to its elements". `slice` used to
// answer no for every `dim > 0`, because it rebuilt the view through the
// shape-derived Tensor constructor, which lays contiguous strides out afresh
// from the new shape. For `dim == 0` that coincides with the truth; for an
// inner axis it describes a compacted tensor that does not exist. Nothing in
// the tree called it that way, so nothing caught it.
//
// Each test therefore addresses the view's elements *through its own reported
// strides* and checks what it finds against the value the parent put there.
// That is the only formulation that fails when the strides are wrong, and it
// is what a downstream kernel would do.

#include <vector>
#include <numeric>
#include <cstdint>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Unit_TensorViewTest : public SushiBLASTest
        {
            protected:
                /**
                 * @brief Copies a whole allocation back to the host as floats.
                 *
                 * The whole allocation, not the view: the point of these tests is to look at
                 * where a view's elements really sit inside the buffer, which means indexing
                 * from the allocation base with the view's own offset and strides.
                 */
                std::vector<float> read_allocation(const Tensor& t, size_t elements)
                {
                    std::vector<float> host(elements);
                    TestEnvironment::get_context().get_queue()
                        .memcpy(host.data(), t.storage->allocation_base(), elements * sizeof(float)).wait();
                    return host;
                }

                /**
                 * @brief Fills a contiguous tensor so that element k holds the value k.
                 *
                 * Makes every element self-identifying: the value read back through a view's
                 * strides *is* the buffer offset it came from, so a stride error shows up as
                 * a wrong number rather than as plausible-looking data.
                 */
                void fill_with_indices(Tensor& t)
                {
                    std::vector<float> host(static_cast<size_t>(t.num_elements));
                    std::iota(host.begin(), host.end(), 0.0f);
                    TestEnvironment::get_context().get_queue()
                        .memcpy(t.data(), host.data(), host.size() * sizeof(float)).wait();
                }

                /**
                 * @brief Asserts that walking @p view by its own strides visits the elements
                 * the parent placed at @p expected_origin plus the slice's offsets.
                 *
                 * @param view The slice under test.
                 * @param buffer The parent allocation, read back to the host.
                 * @param parent_strides The strides of the tensor the slice came from.
                 * @param start Where the slice starts along @p dim, in elements of that axis.
                 * @param dim The axis that was sliced.
                 */
                void expect_view_addresses_its_elements(const Tensor& view,
                                                        const std::vector<float>& buffer,
                                                        const std::array<int64_t, Core::MAX_TENSOR_RANK>& parent_strides,
                                                        int64_t start, int32_t dim)
                {
                    std::array<int64_t, Core::MAX_TENSOR_RANK> idx{};

                    for (int64_t flat = 0; flat < view.num_elements; ++flat)
                    {
                        // Unravel `flat` into the view's logical coordinates, last axis fastest.
                        int64_t rest = flat;
                        for (int d = view.rank - 1; d >= 0; --d)
                        {
                            idx[static_cast<size_t>(d)] = rest % view.shape[d];
                            rest /= view.shape[d];
                        }

                        // Where the view says the element is...
                        int64_t reported = view.storage_offset;
                        for (int d = 0; d < view.rank; ++d)
                            reported += idx[static_cast<size_t>(d)] * view.strides[d];

                        // ...and where it actually is, from the parent's own layout.
                        int64_t truth = 0;
                        for (int d = 0; d < view.rank; ++d)
                        {
                            const int64_t parent_index =
                                idx[static_cast<size_t>(d)] + ((d == dim) ? start : 0);
                            truth += parent_index * parent_strides[static_cast<size_t>(d)];
                        }

                        ASSERT_GE(reported, 0);
                        ASSERT_LT(static_cast<size_t>(reported), buffer.size())
                            << "view element " << flat << " addresses past the end of its allocation";

                        // The parent filled element k with the value k, so the truth offset is
                        // also the value that must be found there.
                        EXPECT_FLOAT_EQ(buffer[static_cast<size_t>(reported)], static_cast<float>(truth))
                            << "slice along dim " << dim << ": logical element " << flat
                            << " is addressed at buffer offset " << reported
                            << " but its value lives at offset " << truth
                            << ". The view's strides do not describe its memory.";
                    }
                }
        };

        // Every axis of a rank-2 row-major tensor. dim 0 always worked; dim 1 is the
        // case that used to report contiguous strides for non-contiguous memory.
        TEST_F(Unit_TensorViewTest, SliceAlongEachAxisAddressesItsElementsRowMajor)
        {
            constexpr int64_t kRows = 4, kCols = 4;
            auto parent = engine->create_tensor({kRows, kCols});
            fill_with_indices(parent);

            const auto buffer = read_allocation(parent, static_cast<size_t>(kRows * kCols));
            const auto parent_strides = parent.strides;

            for (int32_t dim = 0; dim < 2; ++dim)
            {
                const Tensor view = parent.slice(dim, 1, 3);
                ASSERT_EQ(2, view.rank);
                ASSERT_EQ(dim == 0 ? 2 : kRows, view.shape[0]);
                ASSERT_EQ(dim == 1 ? 2 : kCols, view.shape[1]);

                expect_view_addresses_its_elements(view, buffer, parent_strides, 1, dim);
            }
        }

        // The same in column-major, where the strides run the other way and a dim-0
        // slice is the non-contiguous one.
        TEST_F(Unit_TensorViewTest, SliceAlongEachAxisAddressesItsElementsColumnMajor)
        {
            constexpr int64_t kRows = 4, kCols = 4;
            reinit_engine(Core::Layout::COLUMN_MAJOR);

            auto parent = engine->create_tensor({kRows, kCols});
            ASSERT_EQ(Core::Layout::COLUMN_MAJOR, parent.layout);
            fill_with_indices(parent);

            const auto buffer = read_allocation(parent, static_cast<size_t>(kRows * kCols));
            const auto parent_strides = parent.strides;

            for (int32_t dim = 0; dim < 2; ++dim)
            {
                const Tensor view = parent.slice(dim, 1, 3);
                expect_view_addresses_its_elements(view, buffer, parent_strides, 1, dim);
            }
        }

        // Rank 3, so that there is an axis which is neither the outermost nor the
        // innermost - the one a two-dimensional test cannot distinguish.
        TEST_F(Unit_TensorViewTest, SliceAlongEachAxisAddressesItsElementsRankThree)
        {
            constexpr int64_t kB = 3, kM = 4, kN = 5;
            auto parent = engine->create_tensor({kB, kM, kN});
            fill_with_indices(parent);

            const auto buffer = read_allocation(parent, static_cast<size_t>(kB * kM * kN));
            const auto parent_strides = parent.strides;

            for (int32_t dim = 0; dim < 3; ++dim)
            {
                const Tensor view = parent.slice(dim, 1, 3);
                expect_view_addresses_its_elements(view, buffer, parent_strides, 1, dim);
            }
        }

        // A slice keeps its parent's strides, so it reports non-contiguous whenever
        // narrowing an inner axis leaves gaps - and `reshape`, which requires
        // contiguity, refuses it rather than producing a second wrong view.
        TEST_F(Unit_TensorViewTest, InnerAxisSliceReportsItselfNonContiguous)
        {
            auto parent = engine->create_tensor({4, 4});

            const Tensor outer = parent.slice(0, 1, 3);
            EXPECT_TRUE(outer.is_contiguous())
                << "Narrowing the outermost axis of a contiguous row-major tensor leaves it "
                   "contiguous; nothing about its layout changed.";

            const Tensor inner = parent.slice(1, 1, 3);
            EXPECT_FALSE(inner.is_contiguous())
                << "A column slice holds 8 elements spread over 14, so it is not contiguous. "
                   "Reporting otherwise is exactly the lie this fix removed.";

            EXPECT_THROW((void)inner.reshape({8}), std::runtime_error);
        }

        // A whole-axis slice changes nothing, and must therefore leave a tensor that
        // still compares equal to its parent in every respect that matters.
        TEST_F(Unit_TensorViewTest, FullWidthSliceIsTheIdentity)
        {
            auto parent = engine->create_tensor({4, 5});

            for (int32_t dim = 0; dim < 2; ++dim)
            {
                const Tensor view = parent.slice(dim, 0, parent.shape[dim]);
                EXPECT_EQ(parent.shape, view.shape) << "dim " << dim;
                EXPECT_EQ(parent.strides, view.strides) << "dim " << dim;
                EXPECT_EQ(parent.storage_offset, view.storage_offset) << "dim " << dim;
                EXPECT_TRUE(view.is_contiguous()) << "dim " << dim;
            }
        }

        // `region()` is what SushiRuntime's dependency tracker orders on, so it has to
        // cover every byte the view can reach - not merely as many bytes as the view has
        // elements. For a strided view those are different numbers, and the smaller one
        // would leave the tail of the view unguarded.
        TEST_F(Unit_TensorViewTest, RegionCoversEveryByteAStridedViewReaches)
        {
            auto parent = engine->create_tensor({4, 4});
            const Tensor inner = parent.slice(1, 1, 3);

            // Elements at buffer offsets 1,2, 5,6, 9,10, 13,14 - so the view holds 8
            // elements but spans 14.
            EXPECT_EQ(8, inner.num_elements);
            EXPECT_EQ(14, inner.element_span());

            const auto region = inner.region();
            EXPECT_EQ(parent.storage->allocation_base(), region.base);
            EXPECT_EQ(sizeof(float) * 1u, region.offset);
            EXPECT_EQ(sizeof(float) * 14u, region.length)
                << "A region sized from num_elements would stop at offset 9 and leave the "
                   "last three elements of the view unguarded against a concurrent writer.";
        }

        // The contiguous case must not have moved: element_span and num_elements agree
        // there, which is why the distinction went unnoticed for so long.
        TEST_F(Unit_TensorViewTest, ContiguousTensorSpansExactlyItsElements)
        {
            auto parent = engine->create_tensor({4, 5});
            EXPECT_EQ(parent.num_elements, parent.element_span());
            EXPECT_EQ(static_cast<size_t>(parent.num_elements) * sizeof(float), parent.region().length);
        }

        // A transpose swaps strides without moving data, so it too spans its whole
        // matrix - and it always did, which is what makes it a useful control on
        // element_span.
        TEST_F(Unit_TensorViewTest, TransposeSpansItsWholeMatrix)
        {
            auto parent = engine->create_tensor({3, 5});
            const Tensor t = parent.transpose(0, 1);

            EXPECT_EQ(5, t.shape[0]);
            EXPECT_EQ(3, t.shape[1]);
            EXPECT_EQ(parent.num_elements, t.element_span());
        }

        // The stride-explicit constructor is the seam `slice` now goes through, so its
        // own guards matter: they are the last thing standing between a mis-described
        // view and a kernel reading past the end of an allocation.
        TEST_F(Unit_TensorViewTest, StrideExplicitConstructorRejectsIncoherentMetadata)
        {
            auto parent = engine->create_tensor({4, 4});

            using Strides = std::array<int64_t, Core::MAX_TENSOR_RANK>;

            const std::vector<int64_t> dims = {4, 4};
            const Strides contiguous = {4, 1};
            const Strides negative = {4, -1};
            const Strides too_wide = {64, 1};

            EXPECT_NO_THROW((void)Tensor(parent.storage, dims, contiguous,
                                         Core::DataType::FLOAT32, 0, Core::Layout::ROW_MAJOR));

            EXPECT_THROW((void)Tensor(parent.storage, dims, negative,
                                      Core::DataType::FLOAT32, 0, Core::Layout::ROW_MAJOR),
                         std::runtime_error);

            // 4 rows at a stride of 64 reach element 193 of a 16-element allocation. The
            // element count alone (16) would have fitted, which is why the capacity check
            // has to be asked in terms of the span.
            EXPECT_THROW((void)Tensor(parent.storage, dims, too_wide,
                                      Core::DataType::FLOAT32, 0, Core::Layout::ROW_MAJOR),
                         std::runtime_error);
        }

    } // namespace Test
} // namespace SushiBLAS
