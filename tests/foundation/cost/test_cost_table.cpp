/**************************************************************************/
/* test_cost_table.cpp                                                    */
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
 * @file test_cost_table.cpp
 * @brief Tests that the cost table states the cost each rule family promises.
 * @author Mustafa Garip
 * @see tests/foundation/cost/README.md
 */

#include <SushiBLAS/cost/cost_table.hpp>

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /** @brief Holds a convolution shape and the flops the window_dot rule states for it. */
            struct WindowDotRow
            {
                ConvolutionShape shape;   /**< The extents of the convolution. */
                std::uint64_t expected;   /**< Flops, worked by hand from the module README. */
            };

            /** @brief Holds an elementwise operation and the flops it costs for ten elements. */
            struct ElementwiseRow
            {
                SushiRuntime::OperationID id;  /**< The operation. */
                std::uint64_t expected;        /**< Flops for ten elements. */
            };

            /**
             * @brief Returns a dense, a grouped and a depthwise window_dot shape.
             * @details Sixteen outputs of eight channels make 128 outputs per image; each costs two per tap
             * of the channels it sums over: 3 channels and 9 taps, 4 and 9, then 1 and 9.
             */
            [[nodiscard]] std::vector<WindowDotRow> window_dot_rows()
            {
                return {{ConvolutionShape{1, 4, 4, 8, 3, 1, 3, 3, Core::DataType::Float32, 0}, 6912U},
                        {ConvolutionShape{1, 4, 4, 8, 8, 2, 3, 3, Core::DataType::Float32, 0}, 9216U},
                        {ConvolutionShape{1, 4, 4, 8, 8, 8, 3, 3, Core::DataType::Float32, 0}, 2304U}};
            }

            /** @brief Returns an add, a gelu and an exp row with the per-element weight of each. */
            [[nodiscard]] std::vector<ElementwiseRow> elementwise_rows()
            {
                return {{"math.ew.add"_op, 10U}, {"math.nonlinear.gelu"_op, 90U}, {"math.ew.exp"_op, 10U}};
            }

            /** @brief Returns the first row of the table. */
            [[nodiscard]] const CostRow* begin_of(const CostRows& rows)
            {
                return rows.first;
            }

            /** @brief Returns the end of the rows of the table. */
            [[nodiscard]] const CostRow* end_of(const CostRows& rows)
            {
                return rows.first + rows.count;
            }
        } // namespace

        /** @brief Names the fixture of the window_dot rows. */
        class Small_CostTableWindowDot : public ::testing::TestWithParam<WindowDotRow>
        {
        };

        TEST_P(Small_CostTableWindowDot, ConvolutionCostsTwoPerTapOfTheChannelsOfOneGroup)
        {
            const WindowDotRow& row = GetParam();
            EXPECT_EQ(cost_of("spatial.window_dot"_op, row.shape).flops, row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_CostTableWindowDot, ::testing::ValuesIn(window_dot_rows()));

        /** @brief Names the fixture of the elementwise rows. */
        class Small_CostTableElementwise : public ::testing::TestWithParam<ElementwiseRow>
        {
        };

        TEST_P(Small_CostTableElementwise, AnElementwiseRowScalesWithTheCount)
        {
            const ElementwiseRow& row = GetParam();
            EXPECT_EQ(cost_of(row.id, ElementwiseShape{10, Core::DataType::Float32}).flops, row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_CostTableElementwise, ::testing::ValuesIn(elementwise_rows()));

        TEST(Small_CostTable, GemmCostsTwoMNKPerBatch)
        {
            const Cost cost = cost_of("blas.lvl3.gemm"_op, GemmShape{3, 4, 5, 2, Core::DataType::Float32, 0});
            EXPECT_TRUE(cost.is_costed());
            EXPECT_EQ(cost.flops, 240U);
        }

        TEST(Small_CostTable, AGemmShapeLeftAtItsDefaultsIsOneProductWithNoEpilogue)
        {
            GemmShape shape;
            shape.m = 3;
            shape.n = 4;
            shape.k = 5;
            EXPECT_EQ(cost_of("blas.lvl3.gemm"_op, shape).flops, 120U);
        }

        TEST(Small_CostTable, AComplexGemmCostsEightMNK)
        {
            const Cost cost = cost_of("blas.lvl3.gemm"_op, GemmShape{3, 4, 5, 2, Core::DataType::Complex32, 0});
            EXPECT_EQ(cost.flops, 960U);
        }

        TEST(Small_CostTable, AComplexGemmIsJudgedInTheFloat32Class)
        {
            const Cost cost = cost_of("blas.lvl3.gemm"_op, GemmShape{3, 4, 5, 2, Core::DataType::Complex32, 0});
            EXPECT_EQ(cost.dtype, Core::DataType::Float32);
        }

        TEST(Small_CostTable, TheGemmEpilogueAddsItsCountPerElementOfC)
        {
            const Cost cost = cost_of("blas.lvl3.gemm"_op, GemmShape{3, 4, 5, 2, Core::DataType::Float32, 3});
            EXPECT_EQ(cost.flops, 240U + 3U * 3U * 4U * 2U);
        }

        TEST(Small_CostTable, AHalfGemmIsJudgedInTheHalfClass)
        {
            const Cost cost = cost_of("blas.lvl3.gemm"_op, GemmShape{1, 1, 1, 1, Core::DataType::Half, 0});
            EXPECT_EQ(cost.dtype, Core::DataType::Half);
        }

        TEST(Small_CostTable, AConvolutionEpilogueAddsItsCountPerOutput)
        {
            const ConvolutionShape biased{1, 4, 4, 8, 8, 8, 3, 3, Core::DataType::Float32, 1};
            EXPECT_EQ(cost_of("spatial.window_dot"_op, biased).flops, 2U * 16U * 8U * 9U + 16U * 8U);
        }

        TEST(Small_CostTable, TheSumFoldsInputsIntoOutputs)
        {
            EXPECT_EQ(cost_of("math.sum"_op, ReductionShape{10, 1, Core::DataType::Float32}).flops, 9U);
            EXPECT_EQ(cost_of("math.sum"_op, ReductionShape{12, 4, Core::DataType::Float32}).flops, 8U);
        }

        TEST(Small_CostTable, TheMeanAddsOneDivisionPerOutput)
        {
            EXPECT_EQ(cost_of("math.mean"_op, ReductionShape{10, 1, Core::DataType::Float32}).flops, 10U);
        }

        TEST(Small_CostTable, TheAsumAddsOneAbsoluteValuePerElement)
        {
            EXPECT_EQ(cost_of("math.asum"_op, ReductionShape{10, 1, Core::DataType::Float32}).flops, 19U);
        }

        TEST(Small_CostTable, MaxPoolingCostsOneCombinePerTap)
        {
            EXPECT_EQ(cost_of("spatial.pool_max"_op, WindowShape{6, 4, Core::DataType::Float32}).flops, 24U);
        }

        TEST(Small_CostTable, AveragePoolingAlsoCostsPerOutputWork)
        {
            EXPECT_EQ(cost_of("spatial.pool_avg"_op, WindowShape{6, 4, Core::DataType::Float32}).flops, 30U);
        }

        TEST(Small_CostTable, AveragePoolBackwardCostsItsScatter)
        {
            EXPECT_EQ(cost_of("spatial.pool_avg_backward"_op, WindowShape{6, 4, Core::DataType::Float32}).flops,
                      48U);
        }

        TEST(Small_CostTable, MaxPoolBackwardCostsItsScatter)
        {
            EXPECT_EQ(cost_of("spatial.pool_max_backward"_op, WindowShape{6, 4, Core::DataType::Float32}).flops,
                      6U);
        }

        TEST(Small_CostTable, UpsampleBackwardCostsItsScatter)
        {
            EXPECT_EQ(cost_of("spatial.upsample_nearest_backward"_op,
                              ElementwiseShape{10, Core::DataType::Float32}).flops, 10U);
        }

        TEST(Small_CostTable, TheWeightGradientOfAWindowCostsWhatItsForwardDoes)
        {
            const ConvolutionShape depthwise{2, 4, 4, 3, 3, 3, 3, 3, Core::DataType::Float32, 0};
            const Cost forward = cost_of("spatial.window_dot"_op, depthwise);
            const Cost gradient = cost_of("spatial.window_dot_weights_backward"_op, depthwise);
            EXPECT_TRUE(gradient.is_costed());
            EXPECT_EQ(gradient.flops, forward.flops);
        }

        TEST(Small_CostTable, ATransformCostsFiveNLogNPerPowerOfTwoLine)
        {
            const Cost line = cost_of("signal.fft"_op, TransformShape{1, 1, 8, false, Core::DataType::Float32});
            EXPECT_TRUE(line.is_costed());
            EXPECT_EQ(line.flops, 120U);
        }

        TEST(Small_CostTable, ABatchedTwoDimensionalTransformSumsItsRowAndColumnPasses)
        {
            EXPECT_EQ(cost_of("signal.inverse_fft"_op,
                              TransformShape{2, 4, 8, false, Core::DataType::Float32}).flops,
                      1600U);
        }

        TEST(Small_CostTable, ARealEvenRowIsTransformedAsHalfALineAndUntangled)
        {
            EXPECT_EQ(cost_of("signal.fft"_op,
                              TransformShape{1, 1, 8, true, Core::DataType::Float32}).flops, 80U);
        }

        TEST(Small_CostTable, ABluesteinLineCostsThreePaddedPassesAndTheChirp)
        {
            EXPECT_EQ(cost_of("signal.fft"_op,
                              TransformShape{1, 1, 3, false, Core::DataType::Float64}).flops, 330U);
        }

        TEST(Small_CostTable, ACopyCostsNoFlopsAndIsStillCosted)
        {
            const Cost cost = cost_of("spatial.im2col"_op, CopyShape{Core::DataType::Float32});
            EXPECT_TRUE(cost.is_costed());
            EXPECT_EQ(cost.flops, 0U);
        }

        TEST(Small_CostTable, AnUnknownIdIsUncostedNoRule)
        {
            const Cost cost = cost_of("no.such.operation"_op, ElementwiseShape{10, Core::DataType::Float32});
            EXPECT_FALSE(cost.is_costed());
            EXPECT_EQ(cost.reason, UncostedReason::NoRule);
        }

        TEST(Small_CostTable, ARowWithNoStatedShapeIsUncostedNoShape)
        {
            const Cost cost = cost_of("math.ew.add"_op, CostShape{});
            EXPECT_EQ(cost.reason, UncostedReason::NoShape);
        }

        TEST(Small_CostTable, ADeclaredUncostedRowKeepsItsReasonWhateverTheShape)
        {
            const Cost cost = cost_of("index.for_each"_op, ElementwiseShape{10, Core::DataType::Float32});
            EXPECT_EQ(cost.reason, UncostedReason::DataDependent);
            EXPECT_EQ(cost.flops, 0U);
        }

        TEST(Small_CostTable, ANegativeExtentCostsNothing)
        {
            EXPECT_EQ(cost_of("math.ew.add"_op, ElementwiseShape{-5, Core::DataType::Float32}).flops, 0U);
        }

        TEST(Small_CostTable, AnOverflowingProductSaturates)
        {
            const std::int64_t huge = std::int64_t{1} << 62;
            const Cost cost = cost_of("blas.lvl3.gemm"_op,
                                      GemmShape{huge, huge, huge, huge, Core::DataType::Float32, 0});
            EXPECT_EQ(cost.flops, UINT64_MAX);
        }

        TEST(Small_CostTable, TheTableIsNotEmpty)
        {
            EXPECT_GT(cost_rows().count, 0U);
        }

        TEST(Small_CostTable, RowsAreSortedByIdAndNoIdRepeats)
        {
            const CostRows rows = cost_rows();
            const auto not_ascending = [](const CostRow& left, const CostRow& right) { return !(left.id < right.id); };
            EXPECT_EQ(std::adjacent_find(begin_of(rows), end_of(rows), not_ascending), end_of(rows));
        }

        TEST(Small_CostTable, EveryRowIdIsNamed)
        {
            const CostRows rows = cost_rows();
            EXPECT_TRUE(std::all_of(begin_of(rows), end_of(rows), [](const CostRow& row) { return row.id.is_named(); }));
        }

        TEST(Small_CostTable, EveryRowHasAnIdOfItsOwn)
        {
            const CostRows rows = cost_rows();
            std::set<std::uint32_t> values;
            std::for_each(begin_of(rows), end_of(rows), [&values](const CostRow& row) { values.insert(row.id.value); });
            EXPECT_EQ(values.size(), rows.count);
        }

        TEST(Small_CostTable, EveryRowReasonIsKnown)
        {
            const CostRows rows = cost_rows();
            EXPECT_TRUE(std::all_of(begin_of(rows), end_of(rows), [](const CostRow& row) {
                return std::string(to_string(row.reason)) != "unknown";
            }));
        }

        TEST(Small_CostTable, AReasonOutsideTheKnownOnesIsNamedUnknown)
        {
            EXPECT_EQ(std::string(to_string(static_cast<UncostedReason>(255))), "unknown");
        }

        TEST(Small_CostTable, CostOfAgreesWithTheReasonOfEveryRow)
        {
            const CostRows rows = cost_rows();
            const ElementwiseShape shape{4, Core::DataType::Float32};
            EXPECT_TRUE(std::all_of(begin_of(rows), end_of(rows), [&shape](const CostRow& row) {
                const Cost cost = cost_of(row.id, shape);
                return row.reason == UncostedReason::None ? cost.is_costed() : cost.reason == row.reason;
            }));
        }

        TEST(Small_CostTable, FindCostRowReturnsTheRowOfAnId)
        {
            const CostRow* row = find_cost_row("blas.lvl3.gemm"_op);
            ASSERT_NE(row, nullptr);
            EXPECT_EQ(row->id, "blas.lvl3.gemm"_op);
            EXPECT_EQ(find_cost_row("no.such.operation"_op), nullptr);
        }
    } // namespace Test
} // namespace SushiBLAS
