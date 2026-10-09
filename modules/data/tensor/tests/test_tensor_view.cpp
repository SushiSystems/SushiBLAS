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
 * @file test_tensor_view.cpp
 * @brief Tests that a tensor view's offset and strides lead to the elements it occupies.
 * @author Mustafa Garip
 */

#include <vector>
#include <numeric>
#include <cstdint>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

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
                 * @brief Asserts that @p view's own strides lead to the elements its parent placed.
                 * @param buffer The parent allocation, read back to the host.
                 * @param start Where the slice starts along @p dim, in elements of that axis.
                 */
                void expect_view_addresses_its_elements(const Tensor& view,
                                                        const std::vector<float>& buffer,
                                                        const std::array<int64_t, Core::K_MAX_TENSOR_RANK>& parent_strides,
                                                        int64_t start, int32_t dim)
                {
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> idx{};

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

                        // Element k holds k, so the true offset is the expected value.
                        EXPECT_FLOAT_EQ(buffer[static_cast<size_t>(reported)], static_cast<float>(truth))
                            << "slice along dim " << dim << ": logical element " << flat
                            << " is addressed at buffer offset " << reported
                            << " but its value lives at offset " << truth
                            << ". The view's strides do not describe its memory.";
                    }
                }
        };

        /**
         * @brief Tests slices along each axis of a rank-2 row-major tensor.
         * @see tests/unit/README.md
         */
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

        /**
         * @brief Tests slices along each axis of a rank-2 column-major tensor.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_TensorViewTest, SliceAlongEachAxisAddressesItsElementsColumnMajor)
        {
            constexpr int64_t kRows = 4, kCols = 4;
            reinit_engine(Core::Layout::ColumnMajor);

            auto parent = engine->create_tensor({kRows, kCols});
            ASSERT_EQ(Core::Layout::ColumnMajor, parent.layout);
            fill_with_indices(parent);

            const auto buffer = read_allocation(parent, static_cast<size_t>(kRows * kCols));
            const auto parent_strides = parent.strides;

            for (int32_t dim = 0; dim < 2; ++dim)
            {
                const Tensor view = parent.slice(dim, 1, 3);
                expect_view_addresses_its_elements(view, buffer, parent_strides, 1, dim);
            }
        }

        /**
         * @brief Tests slices along each axis of a rank-3 tensor, which has a middle axis.
         * @see tests/unit/README.md
         */
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

        /**
         * @brief Tests that an inner-axis slice reports non-contiguous and reshape refuses it.
         * @see tests/unit/README.md
         */
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

        /**
         * @brief Tests that a whole-axis slice equals its parent.
         * @see tests/unit/README.md
         */
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

        /**
         * @brief Tests that region() covers every byte a strided view reaches.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_TensorViewTest, RegionCoversEveryByteAStridedViewReaches)
        {
            auto parent = engine->create_tensor({4, 4});
            const Tensor inner = parent.slice(1, 1, 3);

            // The view holds 8 elements, at offsets 1,2, 5,6, 9,10, 13,14, and spans 14.
            EXPECT_EQ(8, inner.num_elements);
            EXPECT_EQ(14, inner.element_span());

            const auto region = inner.region();
            EXPECT_EQ(parent.storage->allocation_base(), region.base);
            EXPECT_EQ(sizeof(float) * 1u, region.offset);
            EXPECT_EQ(sizeof(float) * 14u, region.length)
                << "A region sized from num_elements would stop at offset 9 and leave the "
                   "last three elements of the view unguarded against a concurrent writer.";
        }

        /**
         * @brief Tests that a contiguous tensor's element span equals its element count.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_TensorViewTest, ContiguousTensorSpansExactlyItsElements)
        {
            auto parent = engine->create_tensor({4, 5});
            EXPECT_EQ(parent.num_elements, parent.element_span());
            EXPECT_EQ(static_cast<size_t>(parent.num_elements) * sizeof(float), parent.region().length);
        }

        /**
         * @brief Tests that a transposed matrix spans its whole matrix.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_TensorViewTest, TransposeSpansItsWholeMatrix)
        {
            auto parent = engine->create_tensor({3, 5});
            const Tensor t = parent.transpose(0, 1);

            EXPECT_EQ(5, t.shape[0]);
            EXPECT_EQ(3, t.shape[1]);
            EXPECT_EQ(parent.num_elements, t.element_span());
        }

        /**
         * @brief Tests that the stride-explicit constructor rejects incoherent metadata.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_TensorViewTest, StrideExplicitConstructorRejectsIncoherentMetadata)
        {
            auto parent = engine->create_tensor({4, 4});

            using Strides = std::array<int64_t, Core::K_MAX_TENSOR_RANK>;

            const std::vector<int64_t> dims = {4, 4};
            const Strides contiguous = {4, 1};
            const Strides negative = {4, -1};
            const Strides too_wide = {64, 1};

            EXPECT_NO_THROW((void)Tensor(parent.storage, dims, contiguous,
                                         Core::DataType::Float32, 0, Core::Layout::RowMajor));

            EXPECT_THROW((void)Tensor(parent.storage, dims, negative,
                                      Core::DataType::Float32, 0, Core::Layout::RowMajor),
                         std::runtime_error);

            // The span exceeds capacity; the count does not. See tests/unit/README.md.
            EXPECT_THROW((void)Tensor(parent.storage, dims, too_wide,
                                      Core::DataType::Float32, 0, Core::Layout::RowMajor),
                         std::runtime_error);
        }

    } // namespace Test
} // namespace SushiBLAS
