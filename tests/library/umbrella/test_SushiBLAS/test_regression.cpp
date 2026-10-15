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
 * @file test_regression.cpp
 * @brief Tests one fast, deterministic sentinel per fixed defect.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <gtest/gtest.h>
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
        class Medium_Sentinel : public SushiBLASTest {};

        /**
         * @brief Tests that SYRK accepts a HALF operand.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, SyrkAcceptsHalfDtype)
        {
            auto A = engine->create_tensor({4, 3}, Core::DataType::Half);
            auto C = engine->create_tensor({4, 4}, Core::DataType::Half);

            fill_random<sycl::half>(A);
            fill_random<sycl::half>(C);

            engine->set_mode(ExecutionMode::ForceSYCL);
            EXPECT_NO_THROW({
                SushiBLAS::BLASOps(*engine).syrk(A, C, Core::Uplo::Lower,
                                                 Core::Transpose::NoTrans, 1.0, 0.0);
                engine->execute().value();
            }) << "SYRK must accept HALF (SB-2 removed the oneMKL-era restriction).";
        }

        /**
         * @brief Tests that a column-major GEMM indexes A and B by column stride in both modes.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, GemmColumnMajorOperandsUseColumnStrides)
        {
            reinit_engine(Core::Layout::ColumnMajor);

            const std::vector<float> expected{3, 7, 11, 2, 4, 6};

            for (auto mode : {ExecutionMode::ForceSYCL, ExecutionMode::ForceReference})
            {
                SCOPED_TRACE(mode == ExecutionMode::ForceSYCL
                    ? "ForceSYCL: column-major GEMM must index A and B by column stride"
                    : "ForceReference: column-major GEMM must index A and B by column stride");

                auto A = create_initialized_tensor<float>({3, 2}, {1, 3, 5, 2, 4, 6});
                auto B = create_initialized_tensor<float>({2, 2}, {1, 1, 0, 1});
                auto C = engine->create_tensor({3, 2}, Core::DataType::Float32);

                engine->set_mode(mode);
                SushiBLAS::BLASOps(*engine).gemm(A, B, C);
                engine->execute().value();

                EXPECT_TENSOR_NEAR(C, expected, 1e-5);
            }
        }

        /**
         * @brief Tests that re-seeding an engine that has drawn reproduces the stream bit for bit.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, SetSeedRewindsTheRNGStream)
        {
            constexpr size_t kN = 64;

            auto draw = [&](uint64_t seed)
            {
                auto t = engine->create_tensor({static_cast<int64_t>(kN)});
                SushiBLAS::RandomOps(*engine).set_seed(seed);
                SushiBLAS::RandomOps(*engine).uniform(t, 0.0, 1.0);
                engine->execute().value();

                std::vector<float> host(kN);
                auto& q = TestEnvironment::queue();
                q.memcpy(host.data(), t.data(), kN * sizeof(float)).wait();
                return host;
            };

            const auto first = draw(20260803);
            const auto second = draw(20260803);   // same seed, but the engine has now drawn once
            const auto other = draw(20260804);

            EXPECT_EQ(0, std::memcmp(first.data(), second.data(), kN * sizeof(float)))
                << "Re-seeding with the same value must reproduce the stream exactly.";
            EXPECT_NE(0, std::memcmp(first.data(), other.data(), kN * sizeof(float)))
                << "A different seed must still produce a different stream - the reset must "
                   "not have collapsed every seed onto one.";
        }

        /**
         * @brief Tests that fp64 AXPY and SCAL apply their scalar at fp64 precision.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, Fp64Level1ScalarsAreNotNarrowedToFloat)
        {
            constexpr double kAlpha = 1.0 + 0x1p-30;
            ASSERT_EQ(1.0, static_cast<double>(static_cast<float>(kAlpha)))
                << "the test's premise: this alpha is not representable in float.";

            auto& q = TestEnvironment::queue();

            auto x = engine->create_tensor({1}, Core::DataType::Float64);
            auto y = engine->create_tensor({1}, Core::DataType::Float64);
            const double one = 1.0, zero = 0.0;
            q.memcpy(x.data(), &one, sizeof(double)).wait();
            q.memcpy(y.data(), &zero, sizeof(double)).wait();

            SushiBLAS::BLASOps(*engine).axpy(kAlpha, x, y);
            engine->execute().value();

            double got = 0.0;
            q.memcpy(&got, y.data(), sizeof(double)).wait();

            EXPECT_DOUBLE_EQ(kAlpha, got)
                << "an fp64 AXPY must apply alpha at fp64 precision.";
            EXPECT_NE(1.0, got)
                << "alpha was rounded through float somewhere on the way to the kernel.";

            // scal is the same defect in the other Level 1 entry point.
            q.memcpy(x.data(), &one, sizeof(double)).wait();
            SushiBLAS::BLASOps(*engine).scal(kAlpha, x);
            engine->execute().value();
            q.memcpy(&got, x.data(), sizeof(double)).wait();

            EXPECT_DOUBLE_EQ(kAlpha, got) << "an fp64 SCAL must apply alpha at fp64 precision.";
        }

        /**
         * @brief Tests that a batched GEMM reads a distinct slice per batch element in both layouts.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, BatchedGemmAdvancesEveryBatchSlice)
        {
            struct Case
            {
                Core::Layout layout;
                const char* name;
                std::vector<float> a, b, expected;
            };

            const std::vector<Case> cases{
                {Core::Layout::RowMajor, "row-major",
                 {1, 0, 0, 1,   2, 0, 0, 2,   3, 0, 0, 3},
                 {1, 2, 3, 4,   2, 3, 4, 5,   3, 4, 5, 6},
                 {1, 2, 3, 4,   4, 6, 8, 10,  9, 12, 15, 18}},
                {Core::Layout::ColumnMajor, "column-major",
                 {1, 0, 0, 1,   2, 0, 0, 2,   3, 0, 0, 3},
                 {1, 3, 2, 4,   2, 4, 3, 5,   3, 5, 4, 6},
                 {1, 3, 2, 4,   4, 8, 6, 10,  9, 15, 12, 18}},
            };

            for (const auto& c : cases)
            {
                for (auto mode : {ExecutionMode::ForceSYCL, ExecutionMode::ForceReference})
                {
                    SCOPED_TRACE(std::string(c.name) + (mode == ExecutionMode::ForceSYCL ? " / SYCL" : " / reference"));

                    reinit_engine(c.layout);

                    auto A = create_initialized_tensor<float>({3, 2, 2}, c.a);
                    auto B = create_initialized_tensor<float>({3, 2, 2}, c.b);
                    auto C = engine->create_tensor({3, 2, 2}, Core::DataType::Float32);

                    // The batch axis is the slowest-varying one whatever the layout says.
                    EXPECT_EQ(4, A.strides[0]) << "batch stride must be m*k, not 1.";
                    EXPECT_EQ(4, B.strides[0]);
                    EXPECT_EQ(4, C.strides[0]);
                    EXPECT_TRUE(A.is_contiguous()) << "is_contiguous must agree with the layout rule.";

                    engine->set_mode(mode);
                    SushiBLAS::BLASOps(*engine).gemm(A, B, C);
                    engine->execute().value();

                    EXPECT_TENSOR_NEAR(C, c.expected, 1e-4);
                }
            }
        }

        /**
         * @brief Tests that GER and GEMV with a non-zero beta declare their accumulator as read.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, AccumulatingOpsDeclareTheirAccumulatorRead)
        {
            auto contains = [](const std::vector<SushiRuntime::BufferRegion>& set,
                               const SushiRuntime::BufferRegion& r)
            {
                return std::any_of(set.begin(), set.end(),
                                   [&](const auto& e) { return e.base == r.base && e.offset == r.offset; });
            };

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

            {
                GraphProbe probe;
                auto A = engine->create_tensor({2, 3});
                auto x = engine->create_tensor({3});
                auto y = engine->create_tensor({2});

                // beta != 0 -> y is read
                SushiBLAS::BLASOps(probe.recorder()).gemv(A, x, y,
                                                              Core::Transpose::NoTrans, 1.0, 1.0);
                // beta == 0 -> y is not
                SushiBLAS::BLASOps(probe.recorder()).gemv(A, x, y,
                                                              Core::Transpose::NoTrans, 1.0, 0.0);

                ASSERT_EQ(2u, probe.recorded_count());
                const auto tasks = probe.recorded_tasks();
                ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
                EXPECT_TRUE(contains((*tasks)[0].reads, y.region()))
                    << "GEMV with beta != 0 reads y back before overwriting it.";
                EXPECT_TRUE(contains((*tasks)[0].writes, y.region()));
                EXPECT_FALSE(contains((*tasks)[1].reads, y.region()))
                    << "GEMV with beta == 0 fully overwrites y and must stay a pure write - "
                       "declaring a read it does not perform costs the scheduler a false edge.";
                EXPECT_TRUE(contains((*tasks)[1].writes, y.region()));
            }
        }

        /**
         * @brief Tests that a reduction is ordered behind the op that wrote its view of a shared allocation.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, ReductionsOrderAgainstViewsOfOneAllocation)
        {
            GraphProbe probe;

            constexpr int64_t kN = 8;

            // source and packed are disjoint views of one allocation; packed is not at offset zero.
            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Residency::Shared);

            Tensor source(storage, {kN}, Core::DataType::Float32, 0);
            Tensor packed(storage, {kN}, Core::DataType::Float32, kN);
            auto total = engine->create_tensor({1});

            fill_tensor(source, std::vector<float>(static_cast<size_t>(kN), 3.0f));

            // The sum's read of packed must be ordered behind mul's write.
            SushiBLAS::ElementwiseOps(probe.recorder()).mul(source, source, packed);
            SushiBLAS::ReductionOps(probe.recorder()).sum(packed, total);

            // And the same thing behaviourally, for the reader who wants to see it.
            SushiBLAS::ElementwiseOps(*engine).mul(source, source, packed);
            SushiBLAS::ReductionOps(*engine).sum(packed, total);
            engine->execute().value();

            EXPECT_TENSOR_NEAR(total, {static_cast<float>(kN) * 9.0f}, 1e-5);

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                << "The reduction reading `packed` must depend on the op that wrote it. "
                   "An in-degree of 0 means execute_reduction described its operand with "
                   "a bare data() pointer, so the tracker filed the read under a resource "
                   "nothing else touches and emitted no edge at all.";
        }

        namespace
        {
            /**
             * @brief Reads the first FLOAT32 (or complex-real) word of a tensor to the host.
             *
             * The nrm2 sentinels below check values at 1e38 and 1e-40, where the absolute
             * tolerance `verify_tensor` applies is meaningless: 1e-4 of absolute slack
             * passes any answer at all against 2e38, and rejects every answer at 5e-40.
             * They compare relative error instead, so they need the raw number.
             */
            float scalar_of(const Tensor& t)
            {
                float host = 0.0f;
                TestEnvironment::queue()
                    .memcpy(&host, t.data(), sizeof(float)).wait();
                return host;
            }
        }

        /**
         * @brief Tests that both nrm2 entry points stay finite for elements whose squares overflow fp32.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, Nrm2DoesNotOverflowNearTheFloatMaximum)
        {
            // The norm is 2e38, inside fp32's range; 1.2e38 squared is 1.44e76, outside it.
            const std::vector<float> data = {1.2e38f, 1.6e38f};
            const float expected = 2.0e38f;

            auto x = create_initialized_tensor<float>({2}, data);
            auto reduction_result = engine->create_tensor({1});
            auto level1_result = engine->create_tensor({1});

            SushiBLAS::ReductionOps(*engine).nrm2(x, reduction_result);
            SushiBLAS::BLASOps(*engine).nrm2(x, level1_result);
            engine->execute().value();

            const std::pair<const char*, Tensor> answers[] = {
                {"reductions().nrm2", reduction_result},
                {"blas().nrm2", level1_result}};

            for (const auto& [label, result] : answers)
            {
                const float got = scalar_of(result);
                ASSERT_TRUE(std::isfinite(got))
                    << label << " returned " << got << " for a vector whose true norm is "
                    << expected << ". The naive sqrt(sum(x*x)) overflows here; the scaling "
                       "form BLAS specifies does not.";
                EXPECT_NEAR(got / expected, 1.0f, 1e-5f) << label;
            }
        }

        /**
         * @brief Tests that both nrm2 entry points return a non-zero norm for denormal input.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_Sentinel, Nrm2DoesNotUnderflowOnDenormalInput)
        {
            // Both elements are denormal and their norm, 5e-40, is representable.
            const std::vector<float> data = {3.0e-40f, 4.0e-40f};
            const float expected = 5.0e-40f;

            ASSERT_FALSE(std::isnormal(data[0])) << "the test input must actually be denormal";

            auto x = create_initialized_tensor<float>({2}, data);
            auto reduction_result = engine->create_tensor({1});
            auto level1_result = engine->create_tensor({1});

            SushiBLAS::ReductionOps(*engine).nrm2(x, reduction_result);
            SushiBLAS::BLASOps(*engine).nrm2(x, level1_result);
            engine->execute().value();

            const std::pair<const char*, Tensor> answers[] = {
                {"reductions().nrm2", reduction_result},
                {"blas().nrm2", level1_result}};

            for (const auto& [label, result] : answers)
            {
                const float got = scalar_of(result);
                ASSERT_GT(got, 0.0f)
                    << label << " returned " << got << " for a vector whose true norm is "
                    << expected << ". Squaring a denormal underflows to zero; dividing by a "
                       "scale first does not.";

                // The tolerance is 1% because the two entry points round a denormal differently.
                EXPECT_NEAR(got / expected, 1.0f, 1e-2f) << label;
            }
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

            SushiBLAS::BLASOps(*engine).nrm2(x, result);
            engine->execute().value();

            // |3 + 4i| = 5. `abs1` would have given |3| + |4| = 7.
            EXPECT_NEAR(scalar_of(result), 5.0f, 1e-4f)
                << "A complex 2-norm must use the modulus, not BLAS's |Re| + |Im|.";
        }

    } // namespace Test
} // namespace SushiBLAS
