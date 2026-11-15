/**************************************************************************/
/* test_index_along.cpp                                                   */
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
 * @file test_index_along.cpp
 * @brief Tests RowIndexOps::take_along and put_along against a host loop in both execution modes.
 * @author Mustafa Garip
 */

#include <vector>
#include <algorithm>
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

            /// @brief C, the width of every values tensor.
            constexpr int64_t K_COLUMNS = 5;

            /// @brief One position per row, every one inside `[0, C)`.
            const std::vector<int64_t> K_INSIDE = {4, 0, 2, 2, 1, 3};

            /// @brief Positions that hold -1 and C between valid ones.
            const std::vector<int64_t> K_OUTSIDE = {1, -1, 2, K_COLUMNS, 0};

            /// @brief Rows above and below the view in the view test.
            constexpr int64_t K_ROWS_AROUND = 2;

            /** @brief Returns the dense values every typed case starts from, in `[-8, 8]`. */
            std::vector<double> values_for(std::size_t rows)
            {
                return residue_pattern(rows * static_cast<std::size_t>(K_COLUMNS), 7, 17, 8);
            }

            /** @brief Returns one picked value per row, in `[-6, 6]`. */
            std::vector<double> picked_for(std::size_t rows)
            {
                return residue_pattern(rows, 5, 13, 6);
            }
        } // namespace

        /**
         * @brief Runs the element operations in one element type and one index width, skipped
         *        where the device lacks the element type.
         */
        template<typename Case>
        class Medium_RowIndexAlongTyped : public SushiBLASTest
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

                /** @brief Records take_along in a mode; returns picked. */
                std::vector<double> take(const std::vector<double>& values,
                                         const std::vector<int64_t>& positions, ExecutionMode mode)
                {
                    const int64_t count = static_cast<int64_t>(positions.size());
                    const Tensor source = tensor_from<T>(*engine, {count, K_COLUMNS}, values);
                    const Tensor index =
                        tensor_from<Index>(*engine, {count}, as_doubles(positions));
                    Tensor picked = sentinel_tensor<T>(*engine, {count});

                    set_mode(mode);
                    RowIndexOps(recorder()).take_along(source, index, picked);
                    recording->run().value();
                    return values_of<T>(picked);
                }

                /** @brief Records put_along in a mode; returns the values. */
                std::vector<double> put(const std::vector<double>& picked,
                                        const std::vector<int64_t>& positions, ExecutionMode mode)
                {
                    const int64_t count = static_cast<int64_t>(positions.size());
                    const Tensor source = tensor_from<T>(*engine, {count}, picked);
                    const Tensor index =
                        tensor_from<Index>(*engine, {count}, as_doubles(positions));
                    Tensor values = sentinel_tensor<T>(*engine, {count, K_COLUMNS});

                    set_mode(mode);
                    RowIndexOps(recorder()).put_along(source, index, values);
                    recording->run().value();
                    return values_of<T>(values);
                }

                /** @brief Records put_along then take_along on one index; returns the pick. */
                std::vector<double> round_trip(const std::vector<double>& picked,
                                               const std::vector<int64_t>& positions,
                                               ExecutionMode mode)
                {
                    const int64_t count = static_cast<int64_t>(positions.size());
                    const Tensor source = tensor_from<T>(*engine, {count}, picked);
                    const Tensor index =
                        tensor_from<Index>(*engine, {count}, as_doubles(positions));
                    Tensor values = sentinel_tensor<T>(*engine, {count, K_COLUMNS});
                    Tensor again = sentinel_tensor<T>(*engine, {count});

                    set_mode(mode);
                    RowIndexOps(recorder()).put_along(source, index, values);
                    RowIndexOps(recorder()).take_along(values, index, again);
                    recording->run().value();
                    return values_of<T>(again);
                }

                /** @brief Records put_along into the middle rows of a taller parent; returns it. */
                std::vector<double> put_into_rows(const std::vector<double>& picked,
                                                  const std::vector<int64_t>& positions,
                                                  ExecutionMode mode)
                {
                    const int64_t count = static_cast<int64_t>(positions.size());
                    const int64_t total = count + 2 * K_ROWS_AROUND;
                    const Tensor source = tensor_from<T>(*engine, {count}, picked);
                    const Tensor index =
                        tensor_from<Index>(*engine, {count}, as_doubles(positions));
                    const Tensor parent = sentinel_tensor<T>(*engine, {total, K_COLUMNS});
                    Tensor view = matrix_view(parent, count, K_COLUMNS, K_COLUMNS, 1,
                                              K_ROWS_AROUND * K_COLUMNS);

                    set_mode(mode);
                    RowIndexOps(recorder()).put_along(source, index, view);
                    recording->run().value();
                    return values_of<T>(parent);
                }
        };

        using AlongCombos = ::testing::Types<Combo<sycl::half, int32_t>, Combo<sycl::half, int64_t>,
                                             Combo<float, int32_t>, Combo<float, int64_t>,
                                             Combo<double, int32_t>, Combo<double, int64_t>,
                                             Combo<int32_t, int32_t>, Combo<int32_t, int64_t>>;
        SB_TYPED_TEST_SUITE(Medium_RowIndexAlongTyped, AlongCombos);

        TYPED_TEST(Medium_RowIndexAlongTyped, TakeAlongEqualsAHostLoop)
        {
            const std::vector<double> values = values_for(K_INSIDE.size());
            const std::vector<double> expected = host_take_along(values, K_COLUMNS, K_INSIDE);

            EXPECT_EQ(this->take(values, K_INSIDE, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->take(values, K_INSIDE, ExecutionMode::ForceReference), expected);
        }

        TYPED_TEST(Medium_RowIndexAlongTyped, TakeAlongZeroesTheElementOfAnIndexOfMinusOneOrC)
        {
            // Rows 1 and 3 select nothing; their neighbours pick the elements the index names.
            const std::vector<double> values = values_for(K_OUTSIDE.size());
            const std::vector<double> expected = host_take_along(values, K_COLUMNS, K_OUTSIDE);

            EXPECT_EQ(this->take(values, K_OUTSIDE, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->take(values, K_OUTSIDE, ExecutionMode::ForceReference), expected);
        }

        TYPED_TEST(Medium_RowIndexAlongTyped, PutAlongEqualsAHostLoop)
        {
            const std::vector<double> picked = picked_for(K_INSIDE.size());
            const std::vector<double> expected = host_put_along(picked, K_COLUMNS, K_INSIDE);

            EXPECT_EQ(this->put(picked, K_INSIDE, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->put(picked, K_INSIDE, ExecutionMode::ForceReference), expected);
        }

        TYPED_TEST(Medium_RowIndexAlongTyped, PutAlongZeroesTheRowOfAnIndexOfMinusOneOrC)
        {
            // Rows 1 and 3 get no element; their neighbours hold one picked value each.
            const std::vector<double> picked = picked_for(K_OUTSIDE.size());
            const std::vector<double> expected = host_put_along(picked, K_COLUMNS, K_OUTSIDE);

            EXPECT_EQ(this->put(picked, K_OUTSIDE, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->put(picked, K_OUTSIDE, ExecutionMode::ForceReference), expected);
        }

        TYPED_TEST(Medium_RowIndexAlongTyped, TakeAlongAfterPutAlongReturnsThePickedValues)
        {
            const std::vector<double> picked = picked_for(K_INSIDE.size());

            EXPECT_EQ(this->round_trip(picked, K_INSIDE, ExecutionMode::ForceSYCL), picked);
            EXPECT_EQ(this->round_trip(picked, K_INSIDE, ExecutionMode::ForceReference), picked);
        }

        TYPED_TEST(Medium_RowIndexAlongTyped, PutAlongIntoAViewLeavesTheRowsAroundItUnchanged)
        {
            const std::vector<double> picked = picked_for(K_INSIDE.size());

            // The rows above and below the view keep their sentinel; the middle rows are one-hot.
            std::vector<double> expected(
                static_cast<std::size_t>((K_INSIDE.size() + 2 * K_ROWS_AROUND) * K_COLUMNS),
                K_SENTINEL);
            const std::vector<double> middle = host_put_along(picked, K_COLUMNS, K_INSIDE);
            std::copy(middle.begin(), middle.end(), expected.begin() + K_ROWS_AROUND * K_COLUMNS);

            EXPECT_EQ(this->put_into_rows(picked, K_INSIDE, ExecutionMode::ForceSYCL), expected);
            EXPECT_EQ(this->put_into_rows(picked, K_INSIDE, ExecutionMode::ForceReference),
                      expected);
        }

        /** @brief Runs the element operations on FLOAT32 tensors, for cases no type changes. */
        class Medium_RowIndexAlong : public SushiBLASTest
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

        TEST_F(Medium_RowIndexAlong, TakeAlongRefusesValuesOfRankOne)
        {
            const Tensor values = floats({K_COLUMNS});
            Tensor picked = floats({2});

            expect_refusal("index.take_along", "values", [&]
            {
                RowIndexOps(recorder()).take_along(values, positions(2), picked);
            });
        }

        TEST_F(Medium_RowIndexAlong, TakeAlongRefusesAnIndexOfFloatDtype)
        {
            const Tensor values = floats({2, K_COLUMNS});
            Tensor picked = floats({2});

            expect_refusal("index.take_along", "index", [&]
            {
                RowIndexOps(recorder()).take_along(values, floats({2}), picked);
            });
        }

        TEST_F(Medium_RowIndexAlong, TakeAlongRefusesAnIndexWhoseLengthIsNotTheRowCount)
        {
            const Tensor values = floats({3, K_COLUMNS});
            Tensor picked = floats({3});

            expect_refusal("index.take_along", "index", [&]
            {
                RowIndexOps(recorder()).take_along(values, positions(2), picked);
            });
        }

        TEST_F(Medium_RowIndexAlong, TakeAlongRefusesAPickedTensorWhoseLengthIsNotTheRowCount)
        {
            const Tensor values = floats({3, K_COLUMNS});
            Tensor picked = floats({2});

            expect_refusal("index.take_along", "picked", [&]
            {
                RowIndexOps(recorder()).take_along(values, positions(3), picked);
            });
        }

        TEST_F(Medium_RowIndexAlong, TakeAlongRefusesAPickedTensorOfAnotherDtype)
        {
            const Tensor values = floats({2, K_COLUMNS});
            Tensor picked = positions(2);

            expect_refusal("index.take_along", "picked", [&]
            {
                RowIndexOps(recorder()).take_along(values, positions(2), picked);
            });
        }

        TEST_F(Medium_RowIndexAlong, TakeAlongRefusesAPickedTensorThatOverlapsTheValues)
        {
            const Tensor values = floats({2, K_COLUMNS});
            Tensor picked = vector_view(values, 2, 1, 0);

            expect_refusal("index.take_along", "picked", [&]
            {
                RowIndexOps(recorder()).take_along(values, positions(2), picked);
            });
        }

        TEST_F(Medium_RowIndexAlong, PutAlongRefusesPickedValuesOfRankTwo)
        {
            const Tensor picked = floats({2, 1});
            Tensor values = floats({2, K_COLUMNS});

            expect_refusal("index.put_along", "picked", [&]
            {
                RowIndexOps(recorder()).put_along(picked, positions(2), values);
            });
        }

        TEST_F(Medium_RowIndexAlong, PutAlongRefusesAnIndexOfRankTwo)
        {
            const Tensor picked = floats({2});
            const Tensor index = tensor_from<int32_t>(*engine, {2, 1}, {0.0, 1.0});
            Tensor values = floats({2, K_COLUMNS});

            expect_refusal("index.put_along", "index", [&]
            {
                RowIndexOps(recorder()).put_along(picked, index, values);
            });
        }

        TEST_F(Medium_RowIndexAlong, PutAlongRefusesAPickedTensorWhoseLengthIsNotTheRowCount)
        {
            const Tensor picked = floats({2});
            Tensor values = floats({3, K_COLUMNS});

            expect_refusal("index.put_along", "picked", [&]
            {
                RowIndexOps(recorder()).put_along(picked, positions(3), values);
            });
        }

        TEST_F(Medium_RowIndexAlong, PutAlongRefusesValuesOfAnotherDtype)
        {
            const Tensor picked = floats({2});
            Tensor values = tensor_from<int32_t>(*engine, {2, K_COLUMNS},
                                                 std::vector<double>(2 * K_COLUMNS, 0.0));

            expect_refusal("index.put_along", "values", [&]
            {
                RowIndexOps(recorder()).put_along(picked, positions(2), values);
            });
        }

        TEST_F(Medium_RowIndexAlong, PutAlongRefusesValuesWhoseRowsShareOneAddress)
        {
            const Tensor picked = floats({2});
            const Tensor parent = floats({1, K_COLUMNS});
            Tensor values = matrix_view(parent, 2, K_COLUMNS, 0, 1, 0);

            expect_refusal("index.put_along", "values", [&]
            {
                RowIndexOps(recorder()).put_along(picked, positions(2), values);
            });
        }

        TEST_F(Medium_RowIndexAlong, PutAlongRefusesValuesThatOverlapThePickedTensor)
        {
            Tensor values = floats({2, K_COLUMNS});
            const Tensor picked = vector_view(values, 2, 1, 0);

            expect_refusal("index.put_along", "values", [&]
            {
                RowIndexOps(recorder()).put_along(picked, positions(2), values);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
