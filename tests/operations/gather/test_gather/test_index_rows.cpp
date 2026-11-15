/**************************************************************************/
/* test_index_rows.cpp                                                    */
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
 * @file test_index_rows.cpp
 * @brief Tests RowIndexOps::take_rows and add_rows against a host loop in both execution modes.
 * @author Mustafa Garip
 */

#include <limits>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/gather/gather.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "../index_support.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using namespace IndexSupport;

            /// @brief V, the rows of every table.
            constexpr int64_t K_TABLE_ROWS = 6;

            /// @brief D, the width of every table and every rows tensor.
            constexpr int64_t K_WIDTH = 5;

            /// @brief Columns a view skips past its width in the slice test.
            constexpr int64_t K_SLICE_PADDING = 3;

            /// @brief Elements the slice view starts after the first element of its parent.
            constexpr int64_t K_SLICE_OFFSET = 2;

            /// @brief Positions with repeats, so a sum has several terms and a row has none.
            const std::vector<int64_t> K_REPEATED = {3, 0, 5, 3, 1, 2, 0};

            /// @brief Positions that hold -1 and V between valid ones.
            const std::vector<int64_t> K_OUTSIDE = {1, -1, 2, K_TABLE_ROWS, 0};

            /// @brief Positions that pile most rows onto table row 2.
            const std::vector<int64_t> K_PILED = {2, 2, 2, 0, 2, 2, 1, 2};

            /// @brief Returns the dense table every typed case starts from, in `[-8, 8]`.
            std::vector<double> table_values()
            {
                return residue_pattern(static_cast<std::size_t>(K_TABLE_ROWS * K_WIDTH), 7, 17, 8);
            }

            /** @brief Returns the rows for count positions, in `[-6, 6]`. */
            std::vector<double> rows_values(std::size_t count)
            {
                return residue_pattern(count * static_cast<std::size_t>(K_WIDTH), 5, 13, 6);
            }

            /** @brief Returns the elements of a parent that a view of the slice test addresses. */
            std::vector<double> slice_of(const std::vector<double>& parent)
            {
                std::vector<double> view(static_cast<std::size_t>(K_TABLE_ROWS * K_WIDTH));
                for (int64_t v = 0; v < K_TABLE_ROWS; ++v)
                    for (int64_t d = 0; d < K_WIDTH; ++d)
                        view[static_cast<std::size_t>(v * K_WIDTH + d)] =
                            parent[static_cast<std::size_t>(
                                K_SLICE_OFFSET + v * (K_WIDTH + K_SLICE_PADDING) + d)];
                return view;
            }

            /** @brief Returns parent with the elements the slice view addresses replaced. */
            std::vector<double> with_slice(std::vector<double> parent,
                                           const std::vector<double>& view)
            {
                for (int64_t v = 0; v < K_TABLE_ROWS; ++v)
                    for (int64_t d = 0; d < K_WIDTH; ++d)
                        parent[static_cast<std::size_t>(
                            K_SLICE_OFFSET + v * (K_WIDTH + K_SLICE_PADDING) + d)] =
                            view[static_cast<std::size_t>(v * K_WIDTH + d)];
                return parent;
            }
        } // namespace

        /**
         * @brief Runs the row operations in one element type and one index width, skipped where the
         *        device lacks the element type.
         */
        template<typename Case>
        class Medium_RowIndexRowsTyped : public SushiBLASTest
        {
            protected:
                using T = typename Case::Element;
                using Index = typename Case::Index;

                /** @brief Builds the engine and skips the case on a device without T's aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!device_computes<T>())
                        GTEST_SKIP() << "the device does not report the aspect this dtype needs";
                }

                /** @brief Records take_rows in a mode; returns the rows. */
                std::vector<double> take(const std::vector<double>& table,
                                         const std::vector<int64_t>& positions, ExecutionMode mode)
                {
                    const int64_t count = static_cast<int64_t>(positions.size());
                    const Tensor source = tensor_from<T>(*engine, {K_TABLE_ROWS, K_WIDTH}, table);
                    const Tensor index =
                        tensor_from<Index>(*engine, {count}, as_doubles(positions));
                    Tensor rows = sentinel_tensor<T>(*engine, {count, K_WIDTH});

                    set_mode(mode);
                    RowIndexOps(recorder()).take_rows(source, index, rows);
                    recording->run().value();
                    return values_of<T>(rows);
                }
        };

        /** @brief Runs add_rows in one real element type and one index width, skipped as above. */
        template<typename Case>
        class Medium_RowIndexAddTyped : public Medium_RowIndexRowsTyped<Case>
        {
            protected:
                using T = typename Case::Element;
                using Index = typename Case::Index;

                /** @brief Records add_rows of rows into a table and returns the table. */
                std::vector<double> add(const std::vector<double>& rows,
                                        const std::vector<int64_t>& positions,
                                        const std::vector<double>& table, double beta,
                                        ExecutionMode mode)
                {
                    const int64_t count = static_cast<int64_t>(positions.size());
                    const Tensor source = tensor_from<T>(*this->engine, {count, K_WIDTH}, rows);
                    const Tensor index =
                        tensor_from<Index>(*this->engine, {count}, as_doubles(positions));
                    Tensor out = tensor_from<T>(*this->engine, {K_TABLE_ROWS, K_WIDTH}, table);

                    this->set_mode(mode);
                    RowIndexOps(this->recorder()).add_rows(source, index, out, beta);
                    this->recording->run().value();
                    return values_of<T>(out);
                }

                /** @brief Returns the table of `runs` runs of add_rows over fractional rows. */
                std::vector<std::vector<double>> repeated_sums(std::size_t runs, ExecutionMode mode)
                {
                    const std::vector<double> rows = fractional(rows_values(K_PILED.size()));
                    const std::vector<double> table = fractional(table_values());

                    std::vector<std::vector<double>> results;
                    for (std::size_t run = 0; run < runs; ++run)
                        results.push_back(add(rows, K_PILED, table, 1.0, mode));
                    return results;
                }

                /** @brief Records add_rows into a column slice of a wider parent; returns it. */
                std::vector<double> add_into_slice(const std::vector<double>& rows,
                                                   const std::vector<int64_t>& positions,
                                                   const std::vector<double>& parent_values,
                                                   ExecutionMode mode)
                {
                    const int64_t count = static_cast<int64_t>(positions.size());
                    const int64_t row_stride = K_WIDTH + K_SLICE_PADDING;
                    const Tensor source = tensor_from<T>(*this->engine, {count, K_WIDTH}, rows);
                    const Tensor index =
                        tensor_from<Index>(*this->engine, {count}, as_doubles(positions));
                    const Tensor parent =
                        tensor_from<T>(*this->engine, {K_TABLE_ROWS, row_stride}, parent_values);
                    Tensor view = matrix_view(parent, K_TABLE_ROWS, K_WIDTH, row_stride, 1,
                                              K_SLICE_OFFSET);

                    this->set_mode(mode);
                    RowIndexOps(this->recorder()).add_rows(source, index, view, 1.0);
                    this->recording->run().value();
                    return values_of<T>(parent);
                }
        };

        using RowCombos = ::testing::Types<Combo<sycl::half, int32_t>, Combo<sycl::half, int64_t>,
                                           Combo<float, int32_t>, Combo<float, int64_t>,
                                           Combo<double, int32_t>, Combo<double, int64_t>,
                                           Combo<int32_t, int32_t>, Combo<int32_t, int64_t>>;
        SB_TYPED_TEST_SUITE(Medium_RowIndexRowsTyped, RowCombos);

        using RealCombos = ::testing::Types<Combo<sycl::half, int32_t>, Combo<sycl::half, int64_t>,
                                            Combo<float, int32_t>, Combo<float, int64_t>,
                                            Combo<double, int32_t>, Combo<double, int64_t>>;
        SB_TYPED_TEST_SUITE(Medium_RowIndexAddTyped, RealCombos);

        TYPED_TEST(Medium_RowIndexRowsTyped, TakeRowsEqualsAHostLoopWithRepeatedPositions)
        {
            const std::vector<double> table = table_values();
            const std::vector<double> expected =
                host_take_rows(table, K_TABLE_ROWS, K_WIDTH, K_REPEATED);

            EXPECT_EQ(this->take(table, K_REPEATED, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->take(table, K_REPEATED, ExecutionMode::ForceReference), expected);
        }

        TYPED_TEST(Medium_RowIndexRowsTyped, TakeRowsZeroesTheRowOfAnIndexOfMinusOneOrV)
        {
            const std::vector<double> table = table_values();

            // Rows 1 and 3 select nothing; rows 0, 2 and 4 hold table rows 1, 2 and 0.
            const std::vector<double> expected =
                host_take_rows(table, K_TABLE_ROWS, K_WIDTH, K_OUTSIDE);

            EXPECT_EQ(this->take(table, K_OUTSIDE, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->take(table, K_OUTSIDE, ExecutionMode::ForceReference), expected);
        }

        TYPED_TEST(Medium_RowIndexAddTyped, AddRowsWithBetaZeroEqualsTheHostLoopAndSkipsTheTable)
        {
            const std::vector<double> rows = rows_values(K_REPEATED.size());
            const std::vector<double> poisoned(static_cast<std::size_t>(K_TABLE_ROWS * K_WIDTH),
                                               std::numeric_limits<double>::quiet_NaN());
            const std::vector<double> expected =
                host_add_rows(rows, K_WIDTH, K_REPEATED, table_values(), K_TABLE_ROWS, 0.0);

            EXPECT_EQ(this->add(rows, K_REPEATED, poisoned, 0.0, ExecutionMode::ForceSYCL),
                      expected);
            EXPECT_EQ(this->add(rows, K_REPEATED, poisoned, 0.0, ExecutionMode::ForceReference),
                      expected);
        }

        TYPED_TEST(Medium_RowIndexAddTyped, AddRowsWithBetaOneAddsTheRowsToTheTable)
        {
            const std::vector<double> rows = rows_values(K_REPEATED.size());
            const std::vector<double> table = table_values();
            const std::vector<double> expected =
                host_add_rows(rows, K_WIDTH, K_REPEATED, table, K_TABLE_ROWS, 1.0);

            EXPECT_EQ(this->add(rows, K_REPEATED, table, 1.0, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->add(rows, K_REPEATED, table, 1.0, ExecutionMode::ForceReference),
                      expected);
        }

        TYPED_TEST(Medium_RowIndexAddTyped, AddRowsWithBetaHalfScalesTheTableBeforeAdding)
        {
            const std::vector<double> rows = rows_values(K_REPEATED.size());
            const std::vector<double> table = table_values();
            const std::vector<double> expected =
                host_add_rows(rows, K_WIDTH, K_REPEATED, table, K_TABLE_ROWS, 0.5);

            EXPECT_EQ(this->add(rows, K_REPEATED, table, 0.5, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->add(rows, K_REPEATED, table, 0.5, ExecutionMode::ForceReference),
                      expected);
        }

        TYPED_TEST(Medium_RowIndexAddTyped, AddRowsSkipsAnIndexOfMinusOneOrVAndKeepsTheOtherRows)
        {
            // Position 2 is the only valid one, so only table row 2 changes.
            const std::vector<int64_t> positions = {-1, K_TABLE_ROWS, 2};
            const std::vector<double> rows = rows_values(positions.size());
            const std::vector<double> table = table_values();
            const std::vector<double> expected =
                host_add_rows(rows, K_WIDTH, positions, table, K_TABLE_ROWS, 1.0);

            EXPECT_EQ(this->add(rows, positions, table, 1.0, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->add(rows, positions, table, 1.0, ExecutionMode::ForceReference),
                      expected);
        }

        TYPED_TEST(Medium_RowIndexAddTyped, AddRowsOfARepeatedIndexGivesTheSameBytesOnTenRuns)
        {
            const std::vector<std::vector<double>> device =
                this->repeated_sums(10, ExecutionMode::ForceSYCL);
            const std::vector<std::vector<double>> reference =
                this->repeated_sums(10, ExecutionMode::ForceReference);

            EXPECT_EQ(device, std::vector<std::vector<double>>(10, device.front()));
            EXPECT_EQ(reference, std::vector<std::vector<double>>(10, reference.front()));
        }

        TYPED_TEST(Medium_RowIndexAddTyped, AddRowsIntoAColumnSliceLeavesItsParentUnchanged)
        {
            const std::vector<double> parent = residue_pattern(
                static_cast<std::size_t>(K_TABLE_ROWS * (K_WIDTH + K_SLICE_PADDING)), 3, 11, 5);
            const std::vector<double> rows = rows_values(K_REPEATED.size());
            const std::vector<double> expected = with_slice(
                parent, host_add_rows(rows, K_WIDTH, K_REPEATED, slice_of(parent), K_TABLE_ROWS,
                                      1.0));

            EXPECT_EQ(this->add_into_slice(rows, K_REPEATED, parent, ExecutionMode::ForceSYCL),
                      expected);
            EXPECT_EQ(this->add_into_slice(rows, K_REPEATED, parent, ExecutionMode::ForceReference),
                      expected);
        }

        /** @brief Runs the row operations on FLOAT32 tensors, for cases no element type changes. */
        class Medium_RowIndexRows : public SushiBLASTest
        {
            protected:
                /** @brief Returns a FLOAT32 tensor of dims holding K_SENTINEL. */
                Tensor floats(const std::vector<int64_t>& dims)
                {
                    return sentinel_tensor<float>(*engine, dims);
                }

                /** @brief Returns an INT32 index tensor of count zeros. */
                Tensor positions(int64_t count)
                {
                    const std::vector<double> zeros(static_cast<std::size_t>(count), 0.0);
                    return tensor_from<int32_t>(*engine, {count}, zeros);
                }
        };

        TEST_F(Medium_RowIndexRows, TakeRowsReadsATableViewAtANonZeroStorageOffset)
        {
            // The view is the upper half of a table twice as tall; its first row is parent row V.
            const std::vector<double> parent_values = residue_pattern(
                static_cast<std::size_t>(2 * K_TABLE_ROWS * K_WIDTH), 7, 17, 8);
            const Tensor parent = tensor_from<float>(*engine, {2 * K_TABLE_ROWS, K_WIDTH},
                                                     parent_values);
            const Tensor view =
                matrix_view(parent, K_TABLE_ROWS, K_WIDTH, K_WIDTH, 1, K_TABLE_ROWS * K_WIDTH);
            const Tensor index =
                tensor_from<int32_t>(*engine, {static_cast<int64_t>(K_REPEATED.size())},
                                     as_doubles(K_REPEATED));
            Tensor rows = floats({static_cast<int64_t>(K_REPEATED.size()), K_WIDTH});

            const std::vector<double> upper(parent_values.begin() + K_TABLE_ROWS * K_WIDTH,
                                            parent_values.end());

            set_mode(ExecutionMode::ForceSYCL);
            RowIndexOps(recorder()).take_rows(view, index, rows);
            recording->run().value();

            EXPECT_EQ(values_of<float>(rows),
                      host_take_rows(upper, K_TABLE_ROWS, K_WIDTH, K_REPEATED));
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesATableOfRankOne)
        {
            const Tensor table = floats({K_TABLE_ROWS});
            Tensor rows = floats({2, K_WIDTH});

            expect_refusal("index.take_rows", "table", [&]
            {
                RowIndexOps(recorder()).take_rows(table, positions(2), rows);
            });
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesAnIndexOfFloatDtype)
        {
            const Tensor table = floats({K_TABLE_ROWS, K_WIDTH});
            const Tensor index = floats({2});
            Tensor rows = floats({2, K_WIDTH});

            expect_refusal("index.take_rows", "index", [&]
            {
                RowIndexOps(recorder()).take_rows(table, index, rows);
            });
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesAnIndexOfRankTwo)
        {
            const Tensor table = floats({K_TABLE_ROWS, K_WIDTH});
            const Tensor index = tensor_from<int32_t>(*engine, {2, 1}, {0.0, 1.0});
            Tensor rows = floats({2, K_WIDTH});

            expect_refusal("index.take_rows", "index", [&]
            {
                RowIndexOps(recorder()).take_rows(table, index, rows);
            });
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesAnIndexWhoseLengthIsNotTheRowCount)
        {
            const Tensor table = floats({K_TABLE_ROWS, K_WIDTH});
            Tensor rows = floats({3, K_WIDTH});

            expect_refusal("index.take_rows", "index", [&]
            {
                RowIndexOps(recorder()).take_rows(table, positions(2), rows);
            });
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesRowsWhoseWidthIsNotTheTableWidth)
        {
            const Tensor table = floats({K_TABLE_ROWS, K_WIDTH});
            Tensor rows = floats({2, K_WIDTH + 1});

            expect_refusal("index.take_rows", "rows", [&]
            {
                RowIndexOps(recorder()).take_rows(table, positions(2), rows);
            });
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesRowsOfAnotherDtype)
        {
            const Tensor table = floats({K_TABLE_ROWS, K_WIDTH});
            Tensor rows = tensor_from<int32_t>(*engine, {2, K_WIDTH},
                                               std::vector<double>(2 * K_WIDTH, 0.0));

            expect_refusal("index.take_rows", "rows", [&]
            {
                RowIndexOps(recorder()).take_rows(table, positions(2), rows);
            });
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesRowsThatOverlapTheTable)
        {
            Tensor shared = floats({K_TABLE_ROWS, K_WIDTH});

            expect_refusal("index.take_rows", "rows", [&]
            {
                RowIndexOps(recorder()).take_rows(shared, positions(K_TABLE_ROWS), shared);
            });
        }

        TEST_F(Medium_RowIndexRows, TakeRowsRefusesAnElementTypeOfUInt32)
        {
            const Tensor table = engine->create_tensor({K_TABLE_ROWS, K_WIDTH},
                                                       Core::DataType::UInt32);
            Tensor rows = engine->create_tensor({2, K_WIDTH}, Core::DataType::UInt32);

            expect_refusal("index.take_rows", "", [&]
            {
                RowIndexOps(recorder()).take_rows(table, positions(2), rows);
            });
        }

        TEST_F(Medium_RowIndexRows, AddRowsRefusesRowsOfIntegerDtype)
        {
            const Tensor rows = tensor_from<int32_t>(*engine, {2, K_WIDTH},
                                                     std::vector<double>(2 * K_WIDTH, 0.0));
            Tensor table = tensor_from<int32_t>(*engine, {K_TABLE_ROWS, K_WIDTH},
                                                std::vector<double>(K_TABLE_ROWS * K_WIDTH, 0.0));

            expect_refusal("index.add_rows", "", [&]
            {
                RowIndexOps(recorder()).add_rows(rows, positions(2), table);
            });
        }

        TEST_F(Medium_RowIndexRows, AddRowsRefusesATableOfAnotherDtype)
        {
            const Tensor rows = floats({2, K_WIDTH});
            Tensor table = tensor_from<double>(*engine, {K_TABLE_ROWS, K_WIDTH},
                                               std::vector<double>(K_TABLE_ROWS * K_WIDTH, 0.0));

            expect_refusal("index.add_rows", "table", [&]
            {
                RowIndexOps(recorder()).add_rows(rows, positions(2), table);
            });
        }

        TEST_F(Medium_RowIndexRows, AddRowsRefusesAnIndexWhoseLengthIsNotTheRowCount)
        {
            const Tensor rows = floats({3, K_WIDTH});
            Tensor table = floats({K_TABLE_ROWS, K_WIDTH});

            expect_refusal("index.add_rows", "index", [&]
            {
                RowIndexOps(recorder()).add_rows(rows, positions(2), table);
            });
        }

        TEST_F(Medium_RowIndexRows, AddRowsRefusesATableWhoseWidthIsNotTheRowsWidth)
        {
            const Tensor rows = floats({2, K_WIDTH});
            Tensor table = floats({K_TABLE_ROWS, K_WIDTH + 1});

            expect_refusal("index.add_rows", "table", [&]
            {
                RowIndexOps(recorder()).add_rows(rows, positions(2), table);
            });
        }

        TEST_F(Medium_RowIndexRows, AddRowsRefusesATableThatOverlapsTheRows)
        {
            Tensor shared = floats({K_TABLE_ROWS, K_WIDTH});

            expect_refusal("index.add_rows", "table", [&]
            {
                RowIndexOps(recorder()).add_rows(shared, positions(K_TABLE_ROWS), shared);
            });
        }

        TEST_F(Medium_RowIndexRows, AddRowsRefusesATableWhoseRowsShareOneAddress)
        {
            const Tensor rows = floats({2, K_WIDTH});
            const Tensor parent = floats({1, K_WIDTH});
            Tensor table = matrix_view(parent, K_TABLE_ROWS, K_WIDTH, 0, 1, 0);

            expect_refusal("index.add_rows", "table", [&]
            {
                RowIndexOps(recorder()).add_rows(rows, positions(2), table);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
