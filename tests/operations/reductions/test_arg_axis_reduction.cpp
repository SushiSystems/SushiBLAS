/**************************************************************************/
/* test_arg_axis_reduction.cpp                                            */
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
 * @file test_arg_axis_reduction.cpp
 * @brief Tests ReductionOps::argmax_along against a host loop, against argmax and on its refusals.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the name the operation puts in its refusals. */
            constexpr const char* K_OPERATION = "reductions.argmax_along";

            /** @brief Holds a quiet NaN of FLOAT32. */
            constexpr float K_NAN = std::numeric_limits<float>::quiet_NaN();

            /**
             * @brief Returns the lowest index of each segment's maximum, by a host loop.
             * @param axis 0 reduces down the columns, 1 along the rows.
             * @note Valid for input without NaN: `>` keeps the first of equal values.
             */
            std::vector<int64_t> host_argmax_along(const std::vector<float>& matrix, int64_t rows,
                                                   int64_t cols, int axis)
            {
                const int64_t segments = axis == 1 ? rows : cols;
                const int64_t extent = axis == 1 ? cols : rows;
                std::vector<int64_t> indices(static_cast<std::size_t>(segments), 0);

                for (int64_t s = 0; s < segments; ++s)
                {
                    float best = 0.0f;
                    for (int64_t k = 0; k < extent; ++k)
                    {
                        const int64_t flat = axis == 1 ? s * cols + k : k * cols + s;
                        const float value = matrix[static_cast<std::size_t>(flat)];
                        if (k == 0 || value > best)
                        {
                            best = value;
                            indices[static_cast<std::size_t>(s)] = k;
                        }
                    }
                }

                return indices;
            }

            /** @brief Returns integers in `[0, 100]` with repeats, so ties occur in every segment. */
            std::vector<float> repeating_integers(std::size_t count, std::size_t step)
            {
                std::vector<float> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<float>((i * step) % 101);
                return values;
            }
        } // namespace

        class Medium_ArgmaxAlongTest : public SushiBLASTest
        {
            protected:
                /** @brief Runs argmax_along over a rows by cols FLOAT32 matrix in the given mode. */
                std::vector<int64_t> argmax_along(const std::vector<float>& matrix, int64_t rows,
                                                  int64_t cols, int axis, ExecutionMode mode)
                {
                    auto t = create_initialized_tensor<float>({rows, cols}, matrix);
                    auto result = engine->create_tensor({axis == 1 ? rows : cols},
                                                        Core::DataType::Int64);

                    set_mode(mode);
                    SushiBLAS::ReductionOps(recorder()).argmax_along(t, axis, result);
                    recording->run().value();
                    return read_tensor<int64_t>(result);
                }

                /** @brief Runs argmax over a vector and returns its index. */
                int64_t argmax_of(const std::vector<float>& values)
                {
                    auto t = create_initialized_tensor<float>(
                        {static_cast<int64_t>(values.size())}, values);
                    auto result = engine->create_tensor({1}, Core::DataType::Int64);

                    SushiBLAS::ReductionOps(recorder()).argmax(t, result);
                    recording->run().value();
                    return read_tensor<int64_t>(result)[0];
                }

                /** @brief Asserts that call throws std::invalid_argument naming the operation. */
                void expect_refusal(const std::function<void()>& call)
                {
                    try
                    {
                        call();
                        ADD_FAILURE() << K_OPERATION << " accepted an operand it must refuse";
                    }
                    catch (const std::invalid_argument& e)
                    {
                        EXPECT_NE(std::string(e.what()).find(K_OPERATION), std::string::npos)
                            << "the message does not name the operation: " << e.what();
                    }
                }
        };

        TEST_F(Medium_ArgmaxAlongTest, AxisOneGivesTheIndexOfEachRowMaximum)
        {
            const std::vector<float> matrix = repeating_integers(5 * 7, 37);

            EXPECT_EQ(argmax_along(matrix, 5, 7, 1, ExecutionMode::ForceReference),
                      host_argmax_along(matrix, 5, 7, 1));
        }

        TEST_F(Medium_ArgmaxAlongTest, AxisZeroGivesTheIndexOfEachColumnMaximum)
        {
            const std::vector<float> matrix = repeating_integers(6 * 4, 29);

            EXPECT_EQ(argmax_along(matrix, 6, 4, 0, ExecutionMode::ForceReference),
                      host_argmax_along(matrix, 6, 4, 0));
        }

        TEST_F(Medium_ArgmaxAlongTest, ATieInARowPicksTheLowestIndex)
        {
            // The maximum 9 sits at columns 1, 3 and 4 of row 0, and at 2 and 3 of row 1.
            const std::vector<float> matrix = {1, 9, 2, 9, 9,
                                               0, 4, 7, 7, 1};

            EXPECT_EQ(argmax_along(matrix, 2, 5, 1, ExecutionMode::ForceReference),
                      (std::vector<int64_t>{1, 2}));
        }

        TEST_F(Medium_ArgmaxAlongTest, ATieInAColumnPicksTheLowestIndex)
        {
            // Column 0 holds 3 at rows 1 and 2; column 1 holds 8 at rows 0 and 2.
            const std::vector<float> matrix = {1, 8,
                                               3, 2,
                                               3, 8};

            EXPECT_EQ(argmax_along(matrix, 3, 2, 0, ExecutionMode::ForceReference),
                      (std::vector<int64_t>{1, 0}));
        }

        TEST_F(Medium_ArgmaxAlongTest, ANaNAtTheStartOfARowMatchesArgmax)
        {
            const std::vector<float> row = {K_NAN, 4.0f, 9.0f, 1.0f};

            EXPECT_EQ(argmax_along(row, 1, 4, 1, ExecutionMode::ForceReference)[0],
                      argmax_of(row));
        }

        TEST_F(Medium_ArgmaxAlongTest, ANaNAfterTheStartOfARowMatchesArgmax)
        {
            const std::vector<float> row = {2.0f, K_NAN, 9.0f, 9.0f};

            EXPECT_EQ(argmax_along(row, 1, 4, 1, ExecutionMode::ForceReference)[0],
                      argmax_of(row));
        }

        TEST_F(Medium_ArgmaxAlongTest, ARowOfOnlyNaNsMatchesArgmax)
        {
            const std::vector<float> row = {K_NAN, K_NAN, K_NAN};

            EXPECT_EQ(argmax_along(row, 1, 3, 1, ExecutionMode::ForceReference)[0],
                      argmax_of(row));
        }

        TEST_F(Medium_ArgmaxAlongTest, ANaNInAColumnMatchesArgmax)
        {
            const std::vector<float> column = {1.0f, 6.0f, K_NAN, 6.0f};

            EXPECT_EQ(argmax_along(column, 4, 1, 0, ExecutionMode::ForceReference)[0],
                      argmax_of(column));
        }

        TEST_F(Medium_ArgmaxAlongTest, AOneRowMatrixMatchesArgmax)
        {
            const std::vector<float> row = repeating_integers(40, 13);

            EXPECT_EQ(argmax_along(row, 1, 40, 1, ExecutionMode::ForceReference)[0],
                      argmax_of(row));
        }

        TEST_F(Medium_ArgmaxAlongTest, ASegmentOfOneElementGivesIndexZero)
        {
            const std::vector<float> column = {5.0f, -1.0f, 7.0f};

            EXPECT_EQ(argmax_along(column, 3, 1, 1, ExecutionMode::ForceReference),
                      (std::vector<int64_t>{0, 0, 0}));
        }

        TEST_F(Medium_ArgmaxAlongTest, TheDeviceMatchesTheReferenceAlongLongRows)
        {
            // Few long segments take the work-group shape of the segmented fold on a device.
            const std::vector<float> matrix = repeating_integers(3 * 5000, 7);

            EXPECT_EQ(argmax_along(matrix, 3, 5000, 1, ExecutionMode::ForceSYCL),
                      argmax_along(matrix, 3, 5000, 1, ExecutionMode::ForceReference));
        }

        TEST_F(Medium_ArgmaxAlongTest, TheDeviceMatchesTheReferenceAlongLongColumns)
        {
            const std::vector<float> matrix = repeating_integers(5000 * 3, 11);

            EXPECT_EQ(argmax_along(matrix, 5000, 3, 0, ExecutionMode::ForceSYCL),
                      argmax_along(matrix, 5000, 3, 0, ExecutionMode::ForceReference));
        }

        TEST_F(Medium_ArgmaxAlongTest, TheDeviceMatchesAHostLoopAcrossManyShortRows)
        {
            const std::vector<float> matrix = repeating_integers(257 * 9, 31);

            EXPECT_EQ(argmax_along(matrix, 257, 9, 1, ExecutionMode::ForceSYCL),
                      host_argmax_along(matrix, 257, 9, 1));
        }

        TEST_F(Medium_ArgmaxAlongTest, RefusesAnEmptyInput)
        {
            const Tensor empty(SushiRuntime::sushi_ptr<Storage>(), {0, 3});
            auto result = engine->create_tensor({3}, Core::DataType::Int64);

            expect_refusal([&]
            {
                SushiBLAS::ReductionOps(recorder()).argmax_along(empty, 1, result);
            });
        }

        TEST_F(Medium_ArgmaxAlongTest, RefusesAnInputOfTheWrongRank)
        {
            auto t = engine->create_tensor({6});
            auto result = engine->create_tensor({6}, Core::DataType::Int64);

            expect_refusal([&]
            {
                SushiBLAS::ReductionOps(recorder()).argmax_along(t, 0, result);
            });
        }

        TEST_F(Medium_ArgmaxAlongTest, RefusesAnAxisOutsideZeroAndOne)
        {
            auto t = engine->create_tensor({2, 3});
            auto result = engine->create_tensor({2}, Core::DataType::Int64);

            expect_refusal([&]
            {
                SushiBLAS::ReductionOps(recorder()).argmax_along(t, 2, result);
            });
        }

        TEST_F(Medium_ArgmaxAlongTest, RefusesAResultOfTheWrongLength)
        {
            auto t = engine->create_tensor({2, 3});
            auto result = engine->create_tensor({3}, Core::DataType::Int64);

            expect_refusal([&]
            {
                SushiBLAS::ReductionOps(recorder()).argmax_along(t, 1, result);
            });
        }

        TEST_F(Medium_ArgmaxAlongTest, RefusesAResultOfTheWrongRank)
        {
            auto t = engine->create_tensor({2, 3});
            auto result = engine->create_tensor({2, 1}, Core::DataType::Int64);

            expect_refusal([&]
            {
                SushiBLAS::ReductionOps(recorder()).argmax_along(t, 1, result);
            });
        }

        TEST_F(Medium_ArgmaxAlongTest, RefusesAResultThatIsNotInt64)
        {
            auto t = engine->create_tensor({2, 3});
            auto result = engine->create_tensor({2}, Core::DataType::Float32);

            expect_refusal([&]
            {
                SushiBLAS::ReductionOps(recorder()).argmax_along(t, 1, result);
            });
        }

        TEST_F(Medium_ArgmaxAlongTest, RefusesAComplexInput)
        {
            auto t = engine->create_tensor({2, 3}, Core::DataType::Complex32);
            auto result = engine->create_tensor({2}, Core::DataType::Int64);

            expect_refusal([&]
            {
                SushiBLAS::ReductionOps(recorder()).argmax_along(t, 1, result);
            });
        }
    }
}
