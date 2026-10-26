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
 * @file test_cost_table.cpp
 * @brief Tests that the cost table states the cost each rule family promises.
 * @author Mustafa Garip
 * @see tests/foundation/cost/README.md
 */

#include <SushiBLAS/cost/cost_table.hpp>

#include <set>
#include <string>
#include <cstdint>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        TEST(Small_CostTable, GemmCostsTwoMNKPerBatch)
        {
            const Cost cost = cost_of("blas.lvl3.gemm"_op, GemmShape{3, 4, 5, 2, Core::DataType::Float32, 0});
            EXPECT_TRUE(cost.is_costed());
            EXPECT_EQ(cost.flops, 2U * 3U * 4U * 5U * 2U);
            EXPECT_EQ(cost.dtype, Core::DataType::Float32);
        }

        TEST(Small_CostTable, AComplexGemmCostsEightMNK)
        {
            const Cost cost = cost_of("blas.lvl3.gemm"_op, GemmShape{3, 4, 5, 2, Core::DataType::Complex32, 0});
            EXPECT_EQ(cost.flops, 8U * 3U * 4U * 5U * 2U);
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

        TEST(Small_CostTable, ADenseConvolutionCostsTwoPerTapOfEveryInputChannel)
        {
            const ConvolutionShape dense{1, 4, 4, 8, 3, 1, 3, 3, Core::DataType::Float32, 0};
            EXPECT_EQ(cost_of("spatial.window_dot"_op, dense).flops, 2U * 16U * 8U * 3U * 9U);
        }

        TEST(Small_CostTable, AGroupedConvolutionSumsOverTheChannelsOfOneGroup)
        {
            const ConvolutionShape grouped{1, 4, 4, 8, 8, 2, 3, 3, Core::DataType::Float32, 0};
            EXPECT_EQ(cost_of("spatial.window_dot"_op, grouped).flops, 2U * 16U * 8U * 4U * 9U);
        }

        TEST(Small_CostTable, ADepthwiseConvolutionSumsOverTheWindowOnly)
        {
            const ConvolutionShape depthwise{1, 4, 4, 8, 8, 8, 3, 3, Core::DataType::Float32, 0};
            EXPECT_EQ(cost_of("spatial.window_dot"_op, depthwise).flops, 2U * 16U * 8U * 9U);
        }

        TEST(Small_CostTable, AConvolutionEpilogueAddsItsCountPerOutput)
        {
            const ConvolutionShape biased{1, 4, 4, 8, 8, 8, 3, 3, Core::DataType::Float32, 1};
            EXPECT_EQ(cost_of("spatial.window_dot"_op, biased).flops, 2U * 16U * 8U * 9U + 16U * 8U);
        }

        TEST(Small_CostTable, AnElementwiseRowScalesWithTheCount)
        {
            EXPECT_EQ(cost_of("math.ew.add"_op, ElementwiseShape{10, Core::DataType::Float32}).flops, 10U);
            EXPECT_EQ(cost_of("math.nonlinear.gelu"_op, ElementwiseShape{10, Core::DataType::Float32}).flops,
                      90U);
            EXPECT_EQ(cost_of("math.ew.exp"_op, ElementwiseShape{10, Core::DataType::Float32}).flops, 10U);
        }

        TEST(Small_CostTable, AReductionFoldsInputsIntoOutputs)
        {
            const ReductionShape ten_into_one{10, 1, Core::DataType::Float32};
            EXPECT_EQ(cost_of("math.sum"_op, ten_into_one).flops, 9U);
            EXPECT_EQ(cost_of("math.mean"_op, ten_into_one).flops, 10U);
            EXPECT_EQ(cost_of("math.asum"_op, ten_into_one).flops, 19U);
            EXPECT_EQ(cost_of("math.sum"_op, ReductionShape{12, 4, Core::DataType::Float32}).flops, 8U);
        }

        TEST(Small_CostTable, AWindowFoldsItsTapsIntoEachOutput)
        {
            const WindowShape window{6, 4, Core::DataType::Float32};
            EXPECT_EQ(cost_of("spatial.pool_max"_op, window).flops, 24U);
            EXPECT_EQ(cost_of("spatial.pool_avg"_op, window).flops, 30U);
        }

        TEST(Small_CostTable, ThePoolAndUpsampleBackwardsCostTheirScatter)
        {
            const WindowShape window{6, 4, Core::DataType::Float32};
            EXPECT_EQ(cost_of("spatial.pool_avg_backward"_op, window).flops, 48U);
            EXPECT_EQ(cost_of("spatial.pool_max_backward"_op, window).flops, 6U);
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
            const Cost line = cost_of("signal.fft"_op,
                                      TransformShape{1, 1, 8, false, Core::DataType::Float32});
            EXPECT_TRUE(line.is_costed());
            EXPECT_EQ(line.flops, 120U);
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

        TEST(Small_CostTable, RowsAreSortedAndUnique)
        {
            const CostRows rows = cost_rows();
            ASSERT_GT(rows.count, 0U);
            for (std::size_t i = 1; i < rows.count; ++i)
                EXPECT_TRUE(rows.first[i - 1].id < rows.first[i].id) << "row " << i;
        }

        TEST(Small_CostTable, EveryRowNamesAKnownRule)
        {
            std::set<std::uint32_t> seen;
            const CostRows rows = cost_rows();
            for (std::size_t i = 0; i < rows.count; ++i)
            {
                const CostRow& row = rows.first[i];
                EXPECT_TRUE(row.id.is_named());
                EXPECT_TRUE(seen.insert(row.id.value).second);
                EXPECT_NE(std::string(to_string(row.reason)), "unknown");
                if (row.reason == UncostedReason::None)
                    EXPECT_TRUE(cost_of(row.id, ElementwiseShape{4, Core::DataType::Float32}).is_costed());
                else
                    EXPECT_EQ(cost_of(row.id, ElementwiseShape{4, Core::DataType::Float32}).reason,
                              row.reason);
            }
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
