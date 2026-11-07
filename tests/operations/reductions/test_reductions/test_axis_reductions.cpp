/**************************************************************************/
/* test_axis_reductions.cpp                                               */
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
 * @file test_axis_reductions.cpp
 * @brief Tests the axis reductions, the plan they launch from and both of their kernels.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstring>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include "support/tolerance.hpp"
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /**
             * @brief Holds a 3x4 row-major matrix whose reductions are exact in every dtype.
             * @see tests/operations/reductions/README.md
             */
            const std::vector<float> K_ROW_MAJOR =
                {1, -2, 3, 0,
                 5, 4, -1, 2,
                 -3, -7, -2, -9};

            /** @brief The same logical matrix laid out column by column. */
            const std::vector<float> K_COLUMN_MAJOR =
                {1, 5, -3,
                 -2, 4, -7,
                 3, -1, -2,
                 0, 2, -9};

            const std::vector<float> K_ROW_MAX = {3, 5, -2};
            const std::vector<float> K_COL_MAX = {5, 4, 3, 2};
            const std::vector<float> K_ROW_MIN = {-2, -1, -9};
            const std::vector<float> K_COL_MIN = {-3, -7, -2, -9};
            const std::vector<float> K_ROW_SUM = {2, 10, -21};
            const std::vector<float> K_COL_SUM = {3, -5, 0, -7};

            /// @brief The segment count of the inputs the two kernel shapes are driven on.
            constexpr std::size_t K_SEGMENTS = 5;

            /// @brief The extent of each segment; no work-group width divides it.
            constexpr std::size_t K_EXTENT = 733;

            /// @brief The runs a stability case compares byte for byte.
            constexpr int K_RUNS = 16;

            /// @brief The rows of the compensated-sum input.
            constexpr int64_t K_COMPENSATED_ROWS = 16;

            /// @brief The ones that follow the head of each compensated row.
            constexpr int64_t K_COMPENSATED_ONES = 4096;

            /// @brief The head of each compensated row; its ulp in float is 8.
            constexpr float K_COMPENSATED_HEAD = 1e8f;

            /// @brief The gap between a row and the batch maximum that underflows a global shift.
            constexpr float K_LOGIT_GAP = 104.0f;

            /// @brief An extent long enough that few segments of it fill no machine.
            constexpr std::size_t K_LONG_EXTENT = std::size_t{1} << 20;

            /** @brief A device profile with the fields `plan_axis` actually reads. */
            Device::Profile profile(bool is_cpu, std::size_t compute_units,
                                    std::size_t max_work_group, std::size_t sub_group)
            {
                Device::Profile p;
                p.is_cpu = is_cpu;
                p.compute_units = compute_units;
                p.max_work_group_size = max_work_group;
                p.sub_group_size = sub_group;
                return p;
            }

            /** @brief Returns a CPU profile of 12 units and groups up to 8192. */
            Device::Profile cpu_profile()
            {
                return profile(true, 12, 8192, 8);
            }

            /** @brief Returns a GPU profile of 108 units and groups up to 1024. */
            Device::Profile gpu_profile()
            {
                return profile(false, 108, 1024, 32);
            }

            /** @brief Reads a whole tensor back to the host as floats. */
            std::vector<float> host_floats(const Tensor& t)
            {
                std::vector<float> out(static_cast<std::size_t>(t.num_elements));
                TestEnvironment::queue()
                    .memcpy(out.data(), t.data(), out.size() * sizeof(float)).wait();
                return out;
            }

            /** @brief Returns the span of K_SEGMENTS contiguous segments of K_EXTENT elements. */
            Reduce::AxisSpan contiguous_span()
            {
                Reduce::AxisSpan span;
                span.segments = K_SEGMENTS;
                span.extent = K_EXTENT;
                span.segment_stride = K_EXTENT;
                span.element_stride = 1;
                span.out_stride = 1;
                return span;
            }

            /** @brief Returns the group width of the test device, the width of the tree shape. */
            std::size_t tree_width()
            {
                return Reduce::group_width(Device::profile_for(TestEnvironment::queue()));
            }

            /**
             * @brief Folds the K_SEGMENTS segments of host with Accum at the given group width.
             * @param width 1 runs the item-per-segment shape; wider runs the work-group tree.
             * @return One value per segment.
             */
            template<template<typename> class Accum>
            std::vector<float> fold_segments(Engine& engine, std::size_t width,
                                             const std::vector<float>& host)
            {
                auto& q = TestEnvironment::queue();
                auto in = engine.create_tensor({static_cast<int64_t>(K_SEGMENTS),
                                                static_cast<int64_t>(K_EXTENT)});
                auto out = engine.create_tensor({static_cast<int64_t>(K_SEGMENTS)});
                q.memcpy(in.data(), host.data(), host.size() * sizeof(float)).wait();

                const auto identity = [](float x) { return x; };
                const auto keep = [](float v, [[maybe_unused]] std::size_t n) { return v; };

                Reduce::reduce_axis<Accum>(q, in.data_as<float>(), out.data_as<float>(),
                                           contiguous_span(), Reduce::AxisPlan{width}, identity,
                                           keep, {}).wait();
                return host_floats(out);
            }

            /** @brief Returns segments whose values are positive and differ from segment to segment. */
            std::vector<float> graded_segments()
            {
                std::vector<float> host(K_SEGMENTS * K_EXTENT);
                for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                    for (std::size_t k = 0; k < K_EXTENT; ++k)
                        host[s * K_EXTENT + k] =
                            0.1f + static_cast<float>((s * 7 + k) % 31) * 0.031f;
                return host;
            }

            /** @brief Returns segments in which segment s peaks at 100*s + 1, at a position that moves with s. */
            std::vector<float> peaked_segments()
            {
                std::vector<float> host(K_SEGMENTS * K_EXTENT);
                for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                {
                    for (std::size_t k = 0; k < K_EXTENT; ++k)
                        host[s * K_EXTENT + k] = -static_cast<float>(k % 17);
                    host[s * K_EXTENT + (s * 131 + 7) % K_EXTENT] =
                        100.0f * static_cast<float>(s) + 1.0f;
                }
                return host;
            }

            /** @brief Returns the maximum each segment of peaked_segments() holds. */
            std::vector<float> peaks_of_peaked_segments()
            {
                std::vector<float> peaks(K_SEGMENTS);
                for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                    peaks[s] = 100.0f * static_cast<float>(s) + 1.0f;
                return peaks;
            }

            /** @brief Returns segments of varied positive values that sum to a number, not to zero. */
            std::vector<float> stable_segments()
            {
                std::vector<float> host(K_SEGMENTS * K_EXTENT);
                for (std::size_t i = 0; i < host.size(); ++i)
                    host[i] = 0.1f + static_cast<float>(i % 97) * 0.017f;
                return host;
            }

            /** @brief Returns rows of one large head followed by K_COMPENSATED_ONES ones. */
            std::vector<float> head_then_ones()
            {
                const std::size_t row_length = static_cast<std::size_t>(K_COMPENSATED_ONES + 1);
                std::vector<float> host(static_cast<std::size_t>(K_COMPENSATED_ROWS) * row_length,
                                        1.0f);
                for (int64_t r = 0; r < K_COMPENSATED_ROWS; ++r)
                    host[static_cast<std::size_t>(r) * row_length] = K_COMPENSATED_HEAD;
                return host;
            }

            /**
             * @brief Expects every segment sum to lie within n · u · Σ|x| of the sum in double.
             * @param got One value per segment.
             * @param host The K_SEGMENTS segments the values were folded from.
             */
            void expect_segment_sums(const std::vector<float>& got, const std::vector<float>& host)
            {
                ASSERT_EQ(got.size(), K_SEGMENTS);
                for (std::size_t s = 0; s < K_SEGMENTS; ++s)
                {
                    double exact = 0.0;
                    double magnitude = 0.0;
                    for (std::size_t k = 0; k < K_EXTENT; ++k)
                    {
                        const double value = static_cast<double>(host[s * K_EXTENT + k]);
                        exact += value;
                        magnitude += std::fabs(value);
                    }
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<float>(
                        static_cast<double>(got[s]), exact, magnitude, K_EXTENT)))
                        << "segment " << s;
                }
            }

            /**
             * @brief Folds host with a compensated sum on a fresh engine, K_RUNS times.
             * @return The segment sums of each run, in run order.
             */
            std::vector<std::vector<float>> repeated_sums(std::size_t width,
                                                          const std::vector<float>& host)
            {
                std::vector<std::vector<float>> runs;
                for (int r = 0; r < K_RUNS; ++r)
                {
                    Engine fresh(TestEnvironment::runtime());
                    runs.push_back(fold_segments<Reduce::CompensatedSum>(fresh, width, host));
                }
                return runs;
            }

            /** @brief Expects every value to be finite and non-zero, so byte equality is not vacuous. */
            void expect_every_value_finite_and_nonzero(const std::vector<float>& values)
            {
                for (float v : values)
                    EXPECT_TRUE(std::isfinite(v) && v != 0.0f)
                        << "a segment reduced to " << v << ", which is bit-stable for "
                           "reasons unrelated to fold order.";
            }

            /** @brief Expects every run to equal the first byte for byte. */
            void expect_runs_identical(const std::vector<std::vector<float>>& runs)
            {
                for (std::size_t r = 1; r < runs.size(); ++r)
                    EXPECT_EQ(0, std::memcmp(runs.front().data(), runs[r].data(),
                                             runs[r].size() * sizeof(float)))
                        << "run " << r << " differs from run 0. A reduction must be "
                           "reproducible to the bit, not merely to a tolerance.";
            }
        }

        /** @brief One span geometry, the device it is planned for and whether the tree is chosen. */
        struct PlanRow
        {
            const char* name;
            Device::Profile device;
            std::size_t segments;
            std::size_t extent;
            bool cooperative;
        };

        /** @brief Returns the geometries of the plan: enough, few long, few short and empty spans. */
        static std::vector<PlanRow> plan_rows()
        {
            const Device::Profile cpu = cpu_profile();
            const Device::Profile gpu = gpu_profile();
            const std::size_t local = Reduce::group_width(cpu);

            return {
                {"EnoughSegmentsOnACPUTakeOneItemEach", cpu, 12, K_LONG_EXTENT, false},
                {"ManySegmentsOnACPUTakeOneItemEach", cpu, 4096, K_LONG_EXTENT, false},
                // A GPU wants K_WAVES_PER_COMPUTE_UNIT groups per unit before it is full.
                {"EnoughSegmentsOnAGPUTakeOneItemEach", gpu,
                 108 * Device::K_WAVES_PER_COMPUTE_UNIT, K_LONG_EXTENT, false},
                {"FewLongSegmentsOnACPUTakeAWorkGroupEach", cpu, 2, K_LONG_EXTENT, true},
                {"FewLongSegmentsOnAGPUTakeAWorkGroupEach", gpu, 2, K_LONG_EXTENT, true},
                {"FewSegmentsOneShortOfAGroupTakeOneItemEach", cpu, 2, local - 1, false},
                {"FewSegmentsOfAGroupWidthTakeAWorkGroupEach", cpu, 2, local, true},
                {"AnEmptyExtentTakesOneItemEach", cpu, 4, 0, false},
                {"NoSegmentsTakeOneItemEach", cpu, 0, 1024, false},
            };
        }

        /** @brief Names a row by its description, which gtest takes as the test's suffix. */
        template<typename Row>
        static std::string row_name(const ::testing::TestParamInfo<Row>& info)
        {
            return info.param.name;
        }

        /** @brief Runs one geometry of the axis plan. */
        class Small_AxisReductionPlan : public ::testing::TestWithParam<PlanRow>
        {
        };

        /**
         * @brief Tests that the plan picks the shape the geometry calls for.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Small_AxisReductionPlan, TheShapeFollowsTheSegmentGeometry)
        {
            const PlanRow& row = GetParam();

            EXPECT_EQ(Reduce::plan_axis(row.device, row.segments, row.extent).cooperative(),
                      row.cooperative);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_AxisReductionPlan, ::testing::ValuesIn(plan_rows()),
                                 row_name<PlanRow>);

        /** @brief One device whose cooperative plan takes its full group width. */
        struct WidthRow
        {
            const char* name;
            Device::Profile device;
        };

        /** @brief Returns the devices whose few long segments take a work-group each. */
        static std::vector<WidthRow> width_rows()
        {
            return {{"ACPU", cpu_profile()}, {"AGPU", gpu_profile()}};
        }

        /** @brief Runs the cooperative plan of one device. */
        class Small_AxisReductionWidth : public ::testing::TestWithParam<WidthRow>
        {
        };

        /**
         * @brief Tests that a cooperative plan takes the group width of its device.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Small_AxisReductionWidth, ACooperativePlanTakesTheGroupWidthOfItsDevice)
        {
            const Device::Profile& device = GetParam().device;

            EXPECT_EQ(Reduce::plan_axis(device, 2, K_LONG_EXTENT).local,
                      Reduce::group_width(device));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_AxisReductionWidth, ::testing::ValuesIn(width_rows()),
                                 row_name<WidthRow>);

        template<typename T>
        class Medium_AxisReductionsTyped : public SushiBLASTest {};

        using AxisTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Medium_AxisReductionsTyped, AxisTypes);

        TYPED_TEST(Medium_AxisReductionsTyped, RowMaxFoldsAxisOne)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({3}, sushiblas_dtype<T>());
            fill_tensor(x, K_ROW_MAJOR);

            SushiBLAS::ReductionOps(this->recorder()).max_along(x, 1, out);
            this->recording->run().value();

            expect_tensor(out, K_ROW_MAX);
        }

        TYPED_TEST(Medium_AxisReductionsTyped, ColumnMaxFoldsAxisZero)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({4}, sushiblas_dtype<T>());
            fill_tensor(x, K_ROW_MAJOR);

            SushiBLAS::ReductionOps(this->recorder()).max_along(x, 0, out);
            this->recording->run().value();

            expect_tensor(out, K_COL_MAX);
        }

        TYPED_TEST(Medium_AxisReductionsTyped, RowSumFoldsAxisOne)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({3}, sushiblas_dtype<T>());
            fill_tensor(x, K_ROW_MAJOR);

            SushiBLAS::ReductionOps(this->recorder()).sum_along(x, 1, out);
            this->recording->run().value();

            expect_tensor(out, K_ROW_SUM);
        }

        TYPED_TEST(Medium_AxisReductionsTyped, ColumnSumFoldsAxisZero)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({4}, sushiblas_dtype<T>());
            fill_tensor(x, K_ROW_MAJOR);

            SushiBLAS::ReductionOps(this->recorder()).sum_along(x, 0, out);
            this->recording->run().value();

            expect_tensor(out, K_COL_SUM);
        }

        /** @brief Two views of one allocation: a source and a packed tensor behind it. */
        struct ViewsOfOneAllocation
        {
            Tensor source;
            Tensor packed;
        };

        class Medium_AxisReductions : public SushiBLASTest
        {
            protected:
                /** @brief Returns two [3, 4] float views of one allocation, the second at offset 12. */
                ViewsOfOneAllocation views_of_one_allocation()
                {
                    constexpr int64_t kElements = 12;

                    auto storage = make_runtime_storage(TestEnvironment::runtime(),
                        static_cast<size_t>(2 * kElements) * sizeof(float),
                        SushiRuntime::Residency::Shared);

                    return ViewsOfOneAllocation{
                        Tensor(storage, {3, 4}, Core::DataType::Float32, 0),
                        Tensor(storage, {3, 4}, Core::DataType::Float32, kElements)};
                }
        };

        /**
         * @brief Tests that a column-major matrix folds its rows to the row-major maxima.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, ColumnMajorOperandsFoldTheRowsOfTheSameMatrix)
        {
            reinit_engine(Core::Layout::ColumnMajor);

            auto x = engine->create_tensor({3, 4});
            auto rows = engine->create_tensor({3});
            fill_tensor(x, K_COLUMN_MAJOR);

            SushiBLAS::ReductionOps(recorder()).max_along(x, 1, rows);
            recording->run().value();

            expect_tensor(rows, K_ROW_MAX);
        }

        /**
         * @brief Tests that a column-major matrix folds its columns to the row-major sums.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, ColumnMajorOperandsFoldTheColumnsOfTheSameMatrix)
        {
            reinit_engine(Core::Layout::ColumnMajor);

            auto x = engine->create_tensor({3, 4});
            auto cols = engine->create_tensor({4});
            fill_tensor(x, K_COLUMN_MAJOR);

            SushiBLAS::ReductionOps(recorder()).sum_along(x, 0, cols);
            recording->run().value();

            expect_tensor(cols, K_COL_SUM);
        }

        /**
         * @brief Tests that a view at a non-zero storage offset reduces its own elements.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, AViewAtANonZeroOffsetReducesItsOwnElements)
        {
            ViewsOfOneAllocation views = views_of_one_allocation();

            fill_tensor(views.source, std::vector<float>(12, -1.0f));
            fill_tensor(views.packed, K_ROW_MAJOR);

            auto out = engine->create_tensor({3});
            SushiBLAS::ReductionOps(recorder()).max_along(views.packed, 1, out);
            recording->run().value();

            expect_tensor(out, K_ROW_MAX);
        }

        /**
         * @brief Tests that the sum over a view of the product of another view reads the product.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, AxisReductionsOverViewsOfOneAllocationSeeTheirWriter)
        {
            ViewsOfOneAllocation views = views_of_one_allocation();
            auto out = engine->create_tensor({3});
            fill_tensor(views.source, std::vector<float>(12, 3.0f));

            SushiBLAS::ElementwiseOps(recorder()).mul(views.source, views.source, views.packed);
            SushiBLAS::ReductionOps(recorder()).sum_along(views.packed, 1, out);
            recording->run().value();

            expect_tensor_equal(out, {36.0f, 36.0f, 36.0f});
        }

        /**
         * @brief Tests that an axis reduction records a dependency on the op that wrote its input.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, AnAxisReductionIsOrderedBehindTheOpThatWroteItsInput)
        {
            GraphProbe probe;
            ViewsOfOneAllocation views = views_of_one_allocation();
            auto out = engine->create_tensor({3});

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(views.source, views.source,
                                                            views.packed);
            SushiBLAS::ReductionOps(probe.recorder()).sum_along(views.packed, 1, out);

            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            ASSERT_EQ(2u, tasks->size());
            EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                << "The axis reduction reading `packed` must depend on the op that "
                   "wrote it. An in-degree of 0 means the seam described its operand "
                   "with a bare data() pointer instead of Tensor::region().";
        }

        /**
         * @brief Tests that min_along folds the rows of a matrix to their minima.
         */
        TEST_F(Medium_AxisReductions, MinAlongFoldsAxisOne)
        {
            auto matrix = engine->create_tensor({3, 4});
            auto rows = engine->create_tensor({3});
            fill_tensor(matrix, K_ROW_MAJOR);

            SushiBLAS::ReductionOps(recorder()).min_along(matrix, 1, rows);
            recording->run().value();

            expect_tensor(rows, K_ROW_MIN);
        }

        /**
         * @brief Tests that min_along folds the columns of a matrix to their minima.
         */
        TEST_F(Medium_AxisReductions, MinAlongFoldsAxisZero)
        {
            auto matrix = engine->create_tensor({3, 4});
            auto cols = engine->create_tensor({4});
            fill_tensor(matrix, K_ROW_MAJOR);

            SushiBLAS::ReductionOps(recorder()).min_along(matrix, 0, cols);
            recording->run().value();

            expect_tensor(cols, K_COL_MIN);
        }

        /**
         * @brief Tests that the cooperative tree and the item-per-segment fold agree on one input.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, TheCooperativeTreeAgreesWithTheItemPerSegmentFold)
        {
            const std::size_t local = tree_width();
            ASSERT_GT(local, 1u) << "the cooperative shape needs a group wider than one";
            const std::vector<float> host = graded_segments();

            // The two shapes add in different orders. See the test README.
            expect_segment_sums(fold_segments<Reduce::CompensatedSum>(*engine, 1, host), host);
            expect_segment_sums(fold_segments<Reduce::CompensatedSum>(*engine, local, host), host);
        }

        /**
         * @brief Tests that the item-per-segment shape finds the per-segment maximum.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, TheItemPerSegmentShapeFindsTheSegmentMaximum)
        {
            EXPECT_EQ(fold_segments<Reduce::NaNPropagatingMax>(*engine, 1, peaked_segments()),
                      peaks_of_peaked_segments());
        }

        /**
         * @brief Tests that the work-group tree finds the per-segment maximum, bit for bit.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, TheWorkGroupTreeFindsTheSegmentMaximum)
        {
            EXPECT_EQ(fold_segments<Reduce::NaNPropagatingMax>(*engine, tree_width(),
                                                               peaked_segments()),
                      peaks_of_peaked_segments());
        }

        /**
         * @brief Tests that each row sum is Neumaier-compensated.
         *
         * An uncompensated fold returns 1e8 for each row, because every 1.0 falls below the ulp
         * of the running total; 100004096 is a multiple of 8 and so exact in float.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, RowSumsAreCompensated)
        {
            auto x = engine->create_tensor({K_COMPENSATED_ROWS, K_COMPENSATED_ONES + 1});
            const std::vector<float> host = head_then_ones();
            TestEnvironment::queue()
                .memcpy(x.data(), host.data(), host.size() * sizeof(float)).wait();

            auto out = engine->create_tensor({K_COMPENSATED_ROWS});
            SushiBLAS::ReductionOps(recorder()).sum_along(x, 1, out);
            recording->run().value();

            EXPECT_EQ(host_floats(out),
                      std::vector<float>(static_cast<std::size_t>(K_COMPENSATED_ROWS),
                                         K_COMPENSATED_HEAD +
                                             static_cast<float>(K_COMPENSATED_ONES)));
        }

        /**
         * @brief Tests that a row-wise shift by each row's own maximum keeps the row sums finite.
         *
         * Every row is three equal values, so the shifted exponentials sum to exactly 3 whatever
         * the offset of the row, even when the gap to the other row would underflow a global shift.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_AxisReductions, ARowWiseShiftSurvivesAGapThatDestroysAGlobalOne)
        {
            constexpr int64_t kRows = 2;
            constexpr int64_t kCols = 3;

            auto logits = engine->create_tensor({kRows, kCols});
            fill_tensor(logits, {0.0f, 0.0f, 0.0f, -K_LOGIT_GAP, -K_LOGIT_GAP, -K_LOGIT_GAP});
            auto row_max = engine->create_tensor({kRows});
            auto ones_cols = engine->create_tensor({kCols});
            fill_tensor(ones_cols, {1.0f, 1.0f, 1.0f});
            auto shifted = engine->create_tensor({kRows, kCols});
            auto row_sums = engine->create_tensor({kRows});

            SushiBLAS::ReductionOps(recorder()).max_along(logits, 1, row_max);
            SushiBLAS::BLASOps(recorder()).copy(logits, shifted);
            SushiBLAS::BLASOps(recorder()).ger(row_max, ones_cols, shifted, -1.0f);
            SushiBLAS::ElementwiseOps(recorder()).exp(shifted);
            SushiBLAS::ReductionOps(recorder()).sum_along(shifted, 1, row_sums);
            recording->run().value();

            // exp(0) three times, per row.
            expect_tensor_equal(row_sums, {3.0f, 3.0f});
        }

        /** @brief One malformed pair of operands for an axis reduction. */
        struct MalformedRow
        {
            const char* name;
            std::vector<int64_t> operand;
            int axis;
            std::vector<int64_t> result;
            Core::DataType result_dtype;
        };

        /** @brief Returns the malformations an axis reduction refuses at record time. */
        static std::vector<MalformedRow> malformed_rows()
        {
            const Core::DataType f32 = Core::DataType::Float32;
            return {
                // Rank 1 in: there is no axis to fold away.
                {"ARankOneOperand", {4}, 0, {3}, f32},
                // Rank 3 is refused by name, not reduced along the wrong stride.
                {"ARankThreeOperand", {2, 3, 4}, 1, {3}, f32},
                {"AnAxisPastTheRank", {3, 4}, 2, {3}, f32},
                {"ANegativeAxis", {3, 4}, -1, {3}, f32},
                // Folding axis 1 of a [3, 4] leaves 3 values.
                {"AResultTooLongForAxisOne", {3, 4}, 1, {4}, f32},
                {"AResultTooShortForAxisZero", {3, 4}, 0, {3}, f32},
                {"AResultOfTheWrongRank", {3, 4}, 1, {3, 1}, f32},
                {"AResultOfAnotherDtype", {3, 4}, 1, {3}, Core::DataType::Half},
            };
        }

        /** @brief Records an axis reduction on one malformed pair of operands. */
        class Medium_AxisReductionRefusal
            : public SushiBLASTest,
              public ::testing::WithParamInterface<MalformedRow>
        {
            protected:
                /** @brief Returns the row's input operand. */
                Tensor operand() { return engine->create_tensor(GetParam().operand); }

                /** @brief Returns the row's result operand. */
                Tensor result()
                {
                    return engine->create_tensor(GetParam().result, GetParam().result_dtype);
                }
        };

        /**
         * @brief Tests that max_along refuses a malformed operand when it is recorded.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_AxisReductionRefusal, MaxAlongRefusesItAtRecordTime)
        {
            Tensor in = operand();
            Tensor out = result();

            EXPECT_THROW(SushiBLAS::ReductionOps(recorder()).max_along(in, GetParam().axis, out),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that min_along refuses what max_along refuses.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_AxisReductionRefusal, MinAlongRefusesItAtRecordTime)
        {
            Tensor in = operand();
            Tensor out = result();

            EXPECT_THROW(SushiBLAS::ReductionOps(recorder()).min_along(in, GetParam().axis, out),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that sum_along refuses what max_along refuses.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_AxisReductionRefusal, SumAlongRefusesItAtRecordTime)
        {
            Tensor in = operand();
            Tensor out = result();

            EXPECT_THROW(SushiBLAS::ReductionOps(recorder()).sum_along(in, GetParam().axis, out),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_AxisReductionRefusal,
                                 ::testing::ValuesIn(malformed_rows()),
                                 row_name<MalformedRow>);

        /** @brief Names a dtype row by the dtype's name. */
        static std::string dtype_name(const ::testing::TestParamInfo<Core::DataType>& info)
        {
            return Core::to_string(info.param);
        }

        /** @brief Records an axis reduction over one dtype a fold cannot serve. */
        class Medium_AxisReductionDtype
            : public SushiBLASTest,
              public ::testing::WithParamInterface<Core::DataType>
        {
        };

        /**
         * @brief Tests that complex and integer dtypes are refused by name.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_AxisReductionDtype, MaxAlongAndSumAlongRefuseItByName)
        {
            auto x = engine->create_tensor({3, 4}, GetParam());
            auto out = engine->create_tensor({3}, GetParam());

            EXPECT_THROW(SushiBLAS::ReductionOps(recorder()).max_along(x, 1, out),
                         std::invalid_argument);
            EXPECT_THROW(SushiBLAS::ReductionOps(recorder()).sum_along(x, 1, out),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_AxisReductionDtype,
                                 ::testing::Values(Core::DataType::Complex32,
                                                   Core::DataType::Complex64,
                                                   Core::DataType::Int32, Core::DataType::Int64),
                                 dtype_name);

        /** @brief Which of the two kernel shapes a stability case drives. */
        enum class KernelShape : std::uint8_t
        {
            ItemPerSegment,
            WorkGroupTree,
        };

        /** @brief Names a kernel shape for gtest. */
        static std::string shape_name(const ::testing::TestParamInfo<KernelShape>& info)
        {
            return info.param == KernelShape::ItemPerSegment ? "ItemPerSegment" : "WorkGroupTree";
        }

        /** @brief Drives one kernel shape directly, as the planner would not choose it here. */
        class Medium_AxisReductionStability : public ::testing::TestWithParam<KernelShape>
        {
        };

        /**
         * @brief Tests that a kernel shape returns the same bytes on every run.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_AxisReductionStability, AShapeIsBitwiseStable)
        {
            const std::size_t width =
                GetParam() == KernelShape::ItemPerSegment ? std::size_t{1} : tree_width();

            const std::vector<std::vector<float>> runs = repeated_sums(width, stable_segments());

            // A degenerate result is vacuous here. See the test README.
            expect_every_value_finite_and_nonzero(runs.front());
            expect_runs_identical(runs);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_AxisReductionStability,
                                 ::testing::Values(KernelShape::ItemPerSegment,
                                                   KernelShape::WorkGroupTree),
                                 shape_name);

    } // namespace Test
} // namespace SushiBLAS
