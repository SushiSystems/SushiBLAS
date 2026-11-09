/**************************************************************************/
/* test_regression.cpp                                                    */
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
 * @file test_regression.cpp
 * @brief Tests one fast, deterministic sentinel per fixed defect.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cstddef>
#include <complex>
#include <optional>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names one execution mode a sentinel runs under. */
            struct ModeRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The mode the row runs under. */
                ExecutionMode mode;
            };

            /** @brief Returns the two modes that pick a kernel: ForceSYCL and ForceReference. */
            std::vector<ModeRow> kernel_modes()
            {
                return {{"ForceSYCL", ExecutionMode::ForceSYCL},
                        {"ForceReference", ExecutionMode::ForceReference}};
            }

            /** @brief One batched GEMM: a layout, a mode and the hand-computed operands and product. */
            struct BatchedRow
            {
                /** @brief The row name, which gtest takes as the suffix of the test. */
                std::string name;

                /** @brief The layout the engine is rebuilt with. */
                Core::Layout layout;

                /** @brief The mode the row runs under. */
                ExecutionMode mode;

                /** @brief The three A slices, flat. */
                std::vector<float> a;

                /** @brief The three B slices, flat. */
                std::vector<float> b;

                /** @brief The three C slices, flat. */
                std::vector<float> expected;
            };

            /** @brief Returns one row per layout and per kernel mode. */
            std::vector<BatchedRow> batched_rows()
            {
                const std::vector<float> a{1, 0, 0, 1,   2, 0, 0, 2,   3, 0, 0, 3};
                const std::vector<float> row_b{1, 2, 3, 4,   2, 3, 4, 5,   3, 4, 5, 6};
                const std::vector<float> row_c{1, 2, 3, 4,   4, 6, 8, 10,  9, 12, 15, 18};
                const std::vector<float> column_b{1, 3, 2, 4,   2, 4, 3, 5,   3, 5, 4, 6};
                const std::vector<float> column_c{1, 3, 2, 4,   4, 8, 6, 10,  9, 15, 12, 18};

                std::vector<BatchedRow> rows;
                for (const ModeRow& mode : kernel_modes())
                {
                    rows.push_back({"RowMajor_" + mode.name, Core::Layout::RowMajor, mode.mode,
                                    a, row_b, row_c});
                    rows.push_back({"ColumnMajor_" + mode.name, Core::Layout::ColumnMajor,
                                    mode.mode, a, column_b, column_c});
                }
                return rows;
            }

            /** @brief Names a mode row by its name. */
            std::string mode_row_name(const ::testing::TestParamInfo<ModeRow>& info)
            {
                return info.param.name;
            }

            /** @brief Names a batched row by its name. */
            std::string batched_row_name(const ::testing::TestParamInfo<BatchedRow>& info)
            {
                return info.param.name;
            }

            /** @brief Reports whether set holds a region with the base and offset of r. */
            bool contains(const std::vector<SushiRuntime::BufferRegion>& set,
                          const SushiRuntime::BufferRegion& r)
            {
                return std::any_of(set.begin(), set.end(),
                                   [&](const auto& e) { return e.base == r.base && e.offset == r.offset; });
            }

            /** @brief Returns the HALF values of a list of floats. */
            std::vector<sycl::half> halves_of(const std::vector<float>& values)
            {
                std::vector<sycl::half> halves;
                for (const float value : values)
                    halves.push_back(sycl::half(value));
                return halves;
            }

            /** @brief Reads the first FLOAT32 word of a tensor to the host. */
            float scalar_of(const Tensor& t)
            {
                float host = 0.0f;
                TestEnvironment::queue()
                    .memcpy(&host, t.data(), sizeof(float)).wait();
                return host;
            }
        } // namespace

        class Medium_Sentinel : public SushiBLASTest
        {
            protected:
                /** @brief Alpha of the fp64 sentinels: a double holds it, a float rounds it to 1.0. */
                static constexpr double K_WIDE_ALPHA = 1.0 + 0x1p-30;

                /** @brief Returns a FLOAT64 tensor of one element that starts at value. */
                Tensor double_scalar(double value)
                {
                    Tensor t = engine->create_tensor({1}, Core::DataType::Float64);
                    TestEnvironment::queue().memcpy(t.data(), &value, sizeof(double)).wait();
                    return t;
                }

                /** @brief Returns the one FLOAT64 element of t. */
                static double double_of(const Tensor& t)
                {
                    double host = 0.0;
                    TestEnvironment::queue().memcpy(&host, t.data(), sizeof(double)).wait();
                    return host;
                }
        };

        /**
         * @brief Tests that SYRK computes the lower triangle of A * A^T from a HALF operand.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, SyrkAcceptsHalfDtype)
        {
            // Rows (1,0,0), (0,1,0), (0,0,1), (1,1,1): every product is a small integer HALF holds.
            auto A = create_initialized_tensor<sycl::half>(
                {4, 3}, halves_of({1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1}));
            auto C = engine->create_tensor({4, 4}, Core::DataType::Half);
            const float expected[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {1, 1, 1, 3}};

            set_mode(ExecutionMode::ForceSYCL);
            SushiBLAS::BLASOps(recorder()).syrk(A, C, Core::Uplo::Lower,
                                                Core::Transpose::NoTrans, 1.0, 0.0);
            recording->run().value();

            sycl::half host[16];
            TestEnvironment::queue().memcpy(host, C.data(), sizeof(host)).wait();
            for (int row = 0; row < 4; ++row)
            {
                for (int column = 0; column <= row; ++column)
                    EXPECT_EQ(expected[row][column], static_cast<float>(host[row * 4 + column]))
                        << "C(" << row << "," << column << ")";
            }
        }

        /// @brief Runs a sentinel under one kernel mode per row.
        class Medium_SentinelModes : public Medium_Sentinel,
                                     public ::testing::WithParamInterface<ModeRow> {};

        /**
         * @brief Tests that a column-major GEMM indexes A and B by column stride.
         * @see tests/library/umbrella/README.md
         */
        TEST_P(Medium_SentinelModes, GemmColumnMajorOperandsUseColumnStrides)
        {
            reinit_engine(Core::Layout::ColumnMajor);

            auto A = create_initialized_tensor<float>({3, 2}, {1, 3, 5, 2, 4, 6});
            auto B = create_initialized_tensor<float>({2, 2}, {1, 1, 0, 1});
            auto C = engine->create_tensor({3, 2}, Core::DataType::Float32);

            set_mode(GetParam().mode);
            SushiBLAS::BLASOps(recorder()).gemm(A, B, C);
            recording->run().value();

            expect_tensor_equal(C, std::vector<float>{3, 7, 11, 2, 4, 6});
        }

        INSTANTIATE_TEST_SUITE_P(Modes, Medium_SentinelModes,
                                 ::testing::ValuesIn(kernel_modes()), mode_row_name);

        /**
         * @brief Tests that fp64 AXPY applies its scalar at fp64 precision.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, Fp64AxpyIsNotNarrowedToFloat)
        {
            ASSERT_EQ(1.0, static_cast<double>(static_cast<float>(K_WIDE_ALPHA)))
                << "the test's premise: this alpha is not representable in float.";
            auto x = double_scalar(1.0);
            auto y = double_scalar(0.0);

            SushiBLAS::BLASOps(recorder()).axpy(K_WIDE_ALPHA, x, y);
            recording->run().value();

            const double got = double_of(y);
            EXPECT_DOUBLE_EQ(K_WIDE_ALPHA, got)
                << "an fp64 AXPY must apply alpha at fp64 precision.";
            EXPECT_NE(1.0, got)
                << "alpha was rounded through float somewhere on the way to the kernel.";
        }

        /**
         * @brief Tests that fp64 SCAL applies its scalar at fp64 precision.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, Fp64ScalIsNotNarrowedToFloat)
        {
            ASSERT_EQ(1.0, static_cast<double>(static_cast<float>(K_WIDE_ALPHA)))
                << "the test's premise: this alpha is not representable in float.";
            auto x = double_scalar(1.0);

            SushiBLAS::BLASOps(recorder()).scal(K_WIDE_ALPHA, x);
            recording->run().value();

            const double got = double_of(x);
            EXPECT_DOUBLE_EQ(K_WIDE_ALPHA, got) << "an fp64 SCAL must apply alpha at fp64 precision.";
            EXPECT_NE(1.0, got)
                << "alpha was rounded through float somewhere on the way to the kernel.";
        }

        /// @brief Runs the batched GEMM under one layout and one kernel mode per row.
        class Medium_SentinelBatched : public Medium_Sentinel,
                                       public ::testing::WithParamInterface<BatchedRow> {};

        /**
         * @brief Tests that a batched GEMM reads a distinct slice per batch element.
         * @see tests/library/umbrella/README.md
         */
        TEST_P(Medium_SentinelBatched, BatchedGemmAdvancesEveryBatchSlice)
        {
            const BatchedRow& row = GetParam();
            reinit_engine(row.layout);

            auto A = create_initialized_tensor<float>({3, 2, 2}, row.a);
            auto B = create_initialized_tensor<float>({3, 2, 2}, row.b);
            auto C = engine->create_tensor({3, 2, 2}, Core::DataType::Float32);

            // The batch axis is the slowest-varying one whatever the layout says.
            EXPECT_EQ(4, A.strides[0]) << "batch stride must be m*k, not 1.";
            EXPECT_EQ(4, B.strides[0]);
            EXPECT_EQ(4, C.strides[0]);
            EXPECT_TRUE(A.is_contiguous()) << "is_contiguous must agree with the layout rule.";

            set_mode(row.mode);
            SushiBLAS::BLASOps(recorder()).gemm(A, B, C);
            recording->run().value();

            expect_tensor_equal(C, row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(Rows, Medium_SentinelBatched,
                                 ::testing::ValuesIn(batched_rows()), batched_row_name);

        /**
         * @brief Tests that GER declares its accumulator as read and as written.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, GerDeclaresItsAccumulatorRead)
        {
            GraphProbe probe;
            auto x = engine->create_tensor({2});
            auto y = engine->create_tensor({3});
            auto A = engine->create_tensor({2, 3});
            SushiBLAS::BLASOps(probe.recorder()).ger(x, y, A);

            ASSERT_EQ(1u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_TRUE(contains((*tasks)[0].reads, A.region()))
                << "GER accumulates into A: `A = alpha * x * y^H + A` reads it.";
            EXPECT_TRUE(contains((*tasks)[0].writes, A.region()));
        }

        /**
         * @brief Tests that GEMV with a non-zero beta reads y back before it writes it.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, GemvWithBetaDeclaresItsAccumulatorRead)
        {
            GraphProbe probe;
            auto A = engine->create_tensor({2, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({2});

            SushiBLAS::BLASOps(probe.recorder()).gemv(A, x, y, Core::Transpose::NoTrans, 1.0, 1.0);

            ASSERT_EQ(1u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_TRUE(contains((*tasks)[0].reads, y.region()))
                << "GEMV with beta != 0 reads y back before overwriting it.";
            EXPECT_TRUE(contains((*tasks)[0].writes, y.region()));
        }

        /**
         * @brief Tests that GEMV with beta zero stays a pure write of y.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, GemvWithoutBetaStaysAPureWrite)
        {
            GraphProbe probe;
            auto A = engine->create_tensor({2, 3});
            auto x = engine->create_tensor({3});
            auto y = engine->create_tensor({2});

            SushiBLAS::BLASOps(probe.recorder()).gemv(A, x, y, Core::Transpose::NoTrans, 1.0, 0.0);

            ASSERT_EQ(1u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_FALSE(contains((*tasks)[0].reads, y.region()))
                << "GEMV with beta == 0 fully overwrites y and must stay a pure write - "
                   "declaring a read it does not perform costs the scheduler a false edge.";
            EXPECT_TRUE(contains((*tasks)[0].writes, y.region()));
        }

        /// @brief Holds two disjoint views of one allocation, the second not at offset zero.
        class Medium_SentinelSharedViews : public Medium_Sentinel
        {
            protected:
                /** @brief The number of elements of each view. */
                static constexpr int64_t K_N = 8;

                /** @brief Builds the two views over one shared allocation and fills source with 3. */
                void SetUp() override
                {
                    Medium_Sentinel::SetUp();

                    auto storage = make_runtime_storage(TestEnvironment::runtime(),
                        static_cast<size_t>(2 * K_N) * sizeof(float),
                        SushiRuntime::Residency::Shared);

                    source.emplace(storage, SushiRuntime::span<const int64_t>(&K_N, 1),
                                   Core::DataType::Float32, 0);
                    packed.emplace(storage, SushiRuntime::span<const int64_t>(&K_N, 1),
                                   Core::DataType::Float32, K_N);
                    total = engine->create_tensor({1});
                    fill_tensor(*source, std::vector<float>(static_cast<size_t>(K_N), 3.0f));
                }

                /// @brief The producer's input, at offset zero.
                std::optional<Tensor> source;

                /// @brief The producer's output and the reduction's input, past offset zero.
                std::optional<Tensor> packed;

                /// @brief The sum of packed.
                Tensor total;
        };

        /**
         * @brief Tests that a reduction is ordered behind the op that wrote its view of a shared allocation.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_SentinelSharedViews, ReductionsOrderAgainstViewsOfOneAllocation)
        {
            GraphProbe probe;

            // The sum's read of packed must be ordered behind mul's write.
            SushiBLAS::ElementwiseOps(probe.recorder()).mul(*source, *source, *packed);
            SushiBLAS::ReductionOps(probe.recorder()).sum(*packed, total);

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                << "The reduction reading `packed` must depend on the op that wrote it. "
                   "An in-degree of 0 means execute_reduction described its operand with "
                   "a bare data() pointer, so the tracker filed the read under a resource "
                   "nothing else touches and emitted no edge at all.";
        }

        /**
         * @brief Tests that the same pair of ops sums the products once they have run.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_SentinelSharedViews, TheSumOfTheProductsIsTheSumOfTheSquares)
        {
            SushiBLAS::ElementwiseOps(recorder()).mul(*source, *source, *packed);
            SushiBLAS::ReductionOps(recorder()).sum(*packed, total);
            recording->run().value();

            expect_tensor_equal(total, std::vector<float>{static_cast<float>(K_N) * 9.0f});
        }

        /**
         * @brief Tests that Level 1 nrm2 takes the modulus of a complex element.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, Nrm2UsesTheModulusOfAComplexElement)
        {
            const std::vector<std::complex<float>> data = {{3.0f, 4.0f}};

            auto x = create_initialized_tensor<std::complex<float>>({1}, data);
            auto result = engine->create_tensor({1}, Core::DataType::Float32);

            SushiBLAS::BLASOps(recorder()).nrm2(x, result);
            recording->run().value();

            // |3 + 4i| = 5. `abs1` would have given |3| + |4| = 7.
            EXPECT_NEAR(scalar_of(result), 5.0f, absolute_tolerance<float>(128))
                << "A complex 2-norm must use the modulus, not BLAS's |Re| + |Im|.";
        }
    } // namespace Test
} // namespace SushiBLAS
