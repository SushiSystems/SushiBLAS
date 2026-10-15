/**************************************************************************/
/* test_reduce_axis.cpp                                                   */
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
 * @file test_reduce_axis.cpp
 * @brief Tests the axis span, the axis plan and both launch shapes of reduce_axis.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The row count of the matrix the axis reductions fold.
        constexpr std::size_t K_ROWS = 4;

        /// @brief The column count of the matrix the axis reductions fold.
        constexpr std::size_t K_COLUMNS = 300;

        TEST(Unit_ReduceAxis, TotalSegmentsMultipliesBothSegmentCounts)
        {
            Reduce::AxisSpan span;
            span.segments = 3;
            span.segments2 = 5;

            EXPECT_EQ(span.total_segments(), std::size_t{15});
            span.segments2 = 0;
            EXPECT_EQ(span.total_segments(), std::size_t{0});
        }

        TEST(Unit_ReduceAxis, ManySegmentsOrAShortAxisFoldOneSegmentPerItem)
        {
            Device::Profile cpu;
            cpu.is_cpu = true;
            cpu.compute_units = 4;
            cpu.max_work_group_size = 64;

            EXPECT_FALSE(Reduce::plan_axis(cpu, 4, 1000).cooperative());
            EXPECT_FALSE(Reduce::plan_axis(cpu, 2, 10).cooperative());
            EXPECT_FALSE(Reduce::plan_axis(cpu, 0, 1000).cooperative());
            EXPECT_TRUE(Reduce::plan_axis(cpu, 2, 1000).cooperative());
            EXPECT_EQ(Reduce::plan_axis(cpu, 2, 1000).local, Reduce::group_width(cpu));
        }

        /** @brief Runs reduce_axis on the host CPU queue. */
        class Device_ReduceAxis : public HostDeviceTest
        {
            protected:
                /** @brief Returns the row sums of the test matrix under the given plan. */
                std::vector<float> row_sums(const Reduce::AxisPlan& geometry)
                {
                    std::vector<float> values(K_ROWS * K_COLUMNS);
                    for (std::size_t i = 0; i < values.size(); ++i)
                        values[i] = static_cast<float>(i % 11);
                    const SharedArray<float> in(queue(), values);
                    SharedArray<float> out(queue(), K_ROWS);

                    Reduce::AxisSpan span;
                    span.segments = K_ROWS;
                    span.extent = K_COLUMNS;
                    span.segment_stride = K_COLUMNS;
                    span.element_stride = 1;
                    span.out_stride = 1;
                    Reduce::reduce_axis<Reduce::CompensatedSum>(
                        queue(), in.get(), out.get(), span, geometry,
                        [](float x) { return x; }, [](float value, std::size_t) { return value; },
                        {}).wait();
                    return out.values();
                }

                /** @brief Returns the exact row sums of the test matrix. */
                static std::vector<float> expected_row_sums()
                {
                    std::vector<float> sums(K_ROWS, 0.0f);
                    for (std::size_t r = 0; r < K_ROWS; ++r)
                        for (std::size_t c = 0; c < K_COLUMNS; ++c)
                            sums[r] += static_cast<float>((r * K_COLUMNS + c) % 11);
                    return sums;
                }
        };

        TEST_F(Device_ReduceAxis, OneItemPerSegmentGivesTheExactRowSums)
        {
            EXPECT_EQ(row_sums(Reduce::AxisPlan{1}), expected_row_sums());
        }

        TEST_F(Device_ReduceAxis, ACooperativeGroupGivesTheExactRowSums)
        {
            EXPECT_EQ(row_sums(Reduce::AxisPlan{16}), expected_row_sums());
        }

        TEST_F(Device_ReduceAxis, AStridedAxisFoldsTheColumns)
        {
            const SharedArray<double> in(queue(), std::vector<double>{1, 2, 3, 4, 5, 6});
            SharedArray<double> out(queue(), 3);
            Reduce::AxisSpan span;
            span.segments = 3;
            span.extent = 2;
            span.segment_stride = 1;
            span.element_stride = 3;

            Reduce::reduce_axis<Reduce::CompensatedSum>(
                queue(), in.get(), out.get(), span, Reduce::AxisPlan{1},
                [](double x) { return x; },
                [](double value, std::size_t extent)
                {
                    return value / static_cast<double>(extent);
                },
                {}).wait();

            EXPECT_EQ(out.values(), (std::vector<double>{2.5, 3.5, 4.5}));
        }
    } // namespace Test
} // namespace SushiBLAS
