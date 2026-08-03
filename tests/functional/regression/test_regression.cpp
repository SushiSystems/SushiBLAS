/**************************************************************************/
/* test_regression.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// Sentinels for previously fixed bugs — one fast, deterministic test per defect
// so a future change cannot silently reintroduce it.

#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Regression_Sentinel : public SushiBLASTest {};

        // SYRK's old oneMKL-backed implementation threw for sycl::half (oneMKL has
        // no half-precision SYRK). The SB-2 tiling-kernel port lifted that
        // restriction, but test_syrk's own TYPED_TEST suite only covers
        // float/double/complex, so nothing would have caught a regression. This is
        // that sentinel.
        TEST_F(Regression_Sentinel, SyrkAcceptsHalfDtype)
        {
            auto A = engine->create_tensor({4, 3}, Core::DataType::HALF);
            auto C = engine->create_tensor({4, 4}, Core::DataType::HALF);

            fill_random<sycl::half>(A);
            fill_random<sycl::half>(C);

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            EXPECT_NO_THROW({
                engine->blas().syrk(A, C, /*upper=*/false, /*transA=*/false, 1.0, 0.0);
                engine->execute().wait();
            }) << "SYRK must accept HALF (SB-2 removed the oneMKL-era restriction).";
        }

        // GEMM and SYRK used to read a COLUMN_MAJOR A and B with row-major
        // indexing - `a[row*lda + col]` where the standard says `a[row + col*lda]`.
        // Two consequences. The obvious one: `layout` was a no-op for both
        // operands, so a column-major GEMM computed the wrong product. The
        // dangerous one: the driver passes the column-major leading dimension
        // (lda = m for an m x k A), so the wrong index runs past the end of the
        // buffer whenever m > k - the maximum touched index is (m-1)*m + k-1
        // against an m*k allocation.
        //
        // It survived because the reference kernel carried the identical defect,
        // so every differential SYCL-vs-reference test agreed. These are the
        // absolute pins that cannot: hand-computed numbers, both execution modes,
        // and a shape with m > k so an out-of-bounds read is back in range of the
        // sanitizers if it ever returns.
        //
        // A (3x2) = [[1,2],[3,4],[5,6]] column-major is [1,3,5,2,4,6], lda = 3.
        // B (2x2) = [[1,0],[1,1]]       column-major is [1,1,0,1],     ldb = 2.
        // A*B     = [[3,2],[7,4],[11,6]] -> column-major [3,7,11,2,4,6], ldc = 3.
        TEST_F(Regression_Sentinel, GemmColumnMajorOperandsUseColumnStrides)
        {
            reinit_engine(Core::Layout::COLUMN_MAJOR);

            const std::vector<float> expected{3, 7, 11, 2, 4, 6};

            for (auto mode : {ExecutionMode::FORCE_SYCL, ExecutionMode::FORCE_REFERENCE})
            {
                SCOPED_TRACE(mode == ExecutionMode::FORCE_SYCL
                    ? "FORCE_SYCL: column-major GEMM must index A and B by column stride"
                    : "FORCE_REFERENCE: column-major GEMM must index A and B by column stride");

                auto A = create_initialized_tensor<float>({3, 2}, {1, 3, 5, 2, 4, 6});
                auto B = create_initialized_tensor<float>({2, 2}, {1, 1, 0, 1});
                auto C = engine->create_tensor({3, 2}, Core::DataType::FLOAT32);

                engine->set_mode(mode);
                engine->blas().gemm(A, B, C);
                engine->execute().wait();

                EXPECT_TENSOR_NEAR(C, expected, 1e-5);
            }
        }

        // Engine::set_seed() set the seed but left the monotonically increasing RNG
        // offset where the previous draws had pushed it. Every RNG kernel mixes
        // (seed, offset) into Philox, so re-seeding an engine that had already drawn
        // did not reproduce its stream - two identically-seeded generators on one
        // engine produced different data, which is the one thing a seed exists to
        // prevent. Byte equality, not near-equality: a reproduced stream is the same
        // bits or it is not reproduced.
        TEST_F(Regression_Sentinel, SetSeedRewindsTheRngStream)
        {
            constexpr size_t kN = 64;

            auto draw = [&](uint64_t seed)
            {
                auto t = engine->create_tensor({static_cast<int64_t>(kN)});
                engine->random().set_seed(seed);
                engine->random().uniform(t, 0.0, 1.0);
                engine->execute().wait();

                std::vector<float> host(kN);
                auto& q = TestEnvironment::get_context().get_queue();
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

        // axpy/scal/gemv declared their scalars `float` while gemm declared `double`, so
        // an fp64 computation silently received a float-rounded scalar - about 1e-7
        // relative error, no diagnostic, and FLOAT64 only partly true. alpha here is
        // 1 + 2^-30, which double holds exactly and float rounds to exactly 1.0, so the
        // two answers are distinguishable rather than merely close.
        TEST_F(Regression_Sentinel, Fp64Level1ScalarsAreNotNarrowedToFloat)
        {
            constexpr double kAlpha = 1.0 + 0x1p-30;
            ASSERT_EQ(1.0, static_cast<double>(static_cast<float>(kAlpha)))
                << "the test's premise: this alpha is not representable in float.";

            auto& q = TestEnvironment::get_context().get_queue();

            auto x = engine->create_tensor({1}, Core::DataType::FLOAT64);
            auto y = engine->create_tensor({1}, Core::DataType::FLOAT64);
            const double one = 1.0, zero = 0.0;
            q.memcpy(x.data(), &one, sizeof(double)).wait();
            q.memcpy(y.data(), &zero, sizeof(double)).wait();

            engine->blas().axpy(kAlpha, x, y);
            engine->execute().wait();

            double got = 0.0;
            q.memcpy(&got, y.data(), sizeof(double)).wait();

            EXPECT_DOUBLE_EQ(kAlpha, got)
                << "an fp64 AXPY must apply alpha at fp64 precision.";
            EXPECT_NE(1.0, got)
                << "alpha was rounded through float somewhere on the way to the kernel.";

            // scal is the same defect in the other Level 1 entry point.
            q.memcpy(x.data(), &one, sizeof(double)).wait();
            engine->blas().scal(kAlpha, x);
            engine->execute().wait();
            q.memcpy(&got, x.data(), sizeof(double)).wait();

            EXPECT_DOUBLE_EQ(kAlpha, got) << "an fp64 SCAL must apply alpha at fp64 precision.";
        }

        // Tensor's column-major constructor laid out a rank-3 tensor by giving the batch
        // axis stride 1, so the GEMM driver's `str_a = A.strides[rank-3]` came back as 1
        // instead of m*k and every batch element read the same slice. The rule is that a
        // rank-3 tensor's batch axis is outermost and slowest-varying in either layout -
        // `layout` orders the memory inside a slice only.
        //
        // Three distinct batch elements, all three operands varying per batch, so any
        // operand that aliased slice 0 changes the answer:
        //   A_i = (i+1) * I(2)          B_i = {{1,2},{3,4}} + i        C_i = A_i * B_i
        TEST_F(Regression_Sentinel, BatchedGemmAdvancesEveryBatchSlice)
        {
            struct Case
            {
                Core::Layout layout;
                const char* name;
                std::vector<float> a, b, expected;
            };

            const std::vector<Case> cases{
                {Core::Layout::ROW_MAJOR, "row-major",
                 {1, 0, 0, 1,   2, 0, 0, 2,   3, 0, 0, 3},
                 {1, 2, 3, 4,   2, 3, 4, 5,   3, 4, 5, 6},
                 {1, 2, 3, 4,   4, 6, 8, 10,  9, 12, 15, 18}},
                {Core::Layout::COLUMN_MAJOR, "column-major",
                 {1, 0, 0, 1,   2, 0, 0, 2,   3, 0, 0, 3},
                 {1, 3, 2, 4,   2, 4, 3, 5,   3, 5, 4, 6},
                 {1, 3, 2, 4,   4, 8, 6, 10,  9, 15, 12, 18}},
            };

            for (const auto& c : cases)
            {
                for (auto mode : {ExecutionMode::FORCE_SYCL, ExecutionMode::FORCE_REFERENCE})
                {
                    SCOPED_TRACE(std::string(c.name) + (mode == ExecutionMode::FORCE_SYCL ? " / SYCL" : " / reference"));

                    reinit_engine(c.layout);

                    auto A = create_initialized_tensor<float>({3, 2, 2}, c.a);
                    auto B = create_initialized_tensor<float>({3, 2, 2}, c.b);
                    auto C = engine->create_tensor({3, 2, 2}, Core::DataType::FLOAT32);

                    // The batch axis is the slowest-varying one whatever the layout says.
                    EXPECT_EQ(4, A.strides[0]) << "batch stride must be m*k, not 1.";
                    EXPECT_EQ(4, B.strides[0]);
                    EXPECT_EQ(4, C.strides[0]);
                    EXPECT_TRUE(A.is_contiguous()) << "is_contiguous must agree with the layout rule.";

                    engine->set_mode(mode);
                    engine->blas().gemm(A, B, C);
                    engine->execute().wait();

                    EXPECT_TENSOR_NEAR(C, c.expected, 1e-4);
                }
            }
        }

        // ger and gemv(beta != 0) declared their accumulator write-only while reading it.
        // Ordering survived only because the tracker inserts a WAW edge anyway, so this
        // was latent rather than active - and invisible to any behavioural test. It is
        // asserted against the recorded graph instead: take_nodes() hands back what the
        // driver declared.
        TEST_F(Regression_Sentinel, AccumulatingOpsDeclareTheirAccumulatorRead)
        {
            auto contains = [](const std::vector<SushiRuntime::Core::ResourceRegion>& set,
                               const SushiRuntime::Core::ResourceRegion& r)
            {
                return std::any_of(set.begin(), set.end(),
                                   [&](const auto& e) { return e.base == r.base && e.offset == r.offset; });
            };

            {
                auto x = engine->create_tensor({2});
                auto y = engine->create_tensor({3});
                auto A = engine->create_tensor({2, 3});
                engine->blas().ger(x, y, A);

                auto nodes = engine->get_graph().take_nodes();
                ASSERT_EQ(1u, nodes.size());
                EXPECT_TRUE(contains(nodes[0]->io_reads_, A.region()))
                    << "GER accumulates into A: `A = alpha * x * y^H + A` reads it.";
                EXPECT_TRUE(contains(nodes[0]->io_writes_, A.region()));
            }

            {
                auto A = engine->create_tensor({2, 3});
                auto x = engine->create_tensor({3});
                auto y = engine->create_tensor({2});

                engine->blas().gemv(A, x, y, false, 1.0, 1.0);   // beta != 0 -> y is read
                engine->blas().gemv(A, x, y, false, 1.0, 0.0);   // beta == 0 -> y is not

                auto nodes = engine->get_graph().take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_TRUE(contains(nodes[0]->io_reads_, y.region()))
                    << "GEMV with beta != 0 reads y back before overwriting it.";
                EXPECT_TRUE(contains(nodes[0]->io_writes_, y.region()));
                EXPECT_FALSE(contains(nodes[1]->io_reads_, y.region()))
                    << "GEMV with beta == 0 fully overwrites y and must stay a pure write - "
                       "declaring a read it does not perform costs the scheduler a false edge.";
                EXPECT_TRUE(contains(nodes[1]->io_writes_, y.region()));
            }
        }

        // Dispatch::execute_reduction described its operands to the dependency tracker
        // as bare `data()` pointers rather than `Tensor::region()`. Every other op
        // family had been converted; this one was missed, and for a tensor at offset
        // zero the two are indistinguishable, so nothing caught it.
        //
        // For a view inside a shared arena they are not the same thing at all. A
        // ResourceRegion built from a bare pointer keys on THAT pointer as its
        // allocation base; a region from Tensor::region() keys on the allocation base
        // and carries the view's byte interval. Two views of one arena therefore got
        // two different bases, ResourceRegion::overlaps returned false, and the tracker
        // emitted no edge at all - not a wrong edge, none. The reduction was then free
        // to run before the op that produced its input.
        //
        // Not theoretical: SushiAI's first arena-packed training run reported a loss of
        // -57 where ln 6 was expected, because reductions().max raced the bias_add
        // feeding it and blas().scal beat reductions().sum to the loss.
        //
        // Asserted on the recorded graph rather than on the answer, because the answer
        // is exactly what a race makes unreliable: with no edge the two tasks MAY still
        // come out in the right order by luck, and a test that passes by luck guards
        // nothing. The in-degree is the fact that decides it.
        TEST_F(Regression_Sentinel, ReductionsOrderAgainstViewsOfOneAllocation)
        {
            constexpr int64_t kN = 8;

            // One allocation, two disjoint views - the shape a memory planner produces.
            // The consumer's operand is not at offset zero, which is what makes a
            // bare-pointer region and a real region genuinely disagree.
            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());

            Tensor source(storage, {kN}, Core::DataType::FLOAT32, 0);
            Tensor packed(storage, {kN}, Core::DataType::FLOAT32, kN);
            auto total = engine->create_tensor({1});

            fill_tensor(source, std::vector<float>(static_cast<size_t>(kN), 3.0f));

            // mul writes `packed`; sum then reads it. The read must be ordered behind
            // the write.
            engine->elementwise().mul(source, source, packed);
            engine->reductions().sum(packed, total);

            auto nodes = engine->get_graph().take_nodes();
            ASSERT_EQ(2u, nodes.size());
            EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                << "The reduction reading `packed` must depend on the op that wrote it. "
                   "An in-degree of 0 means execute_reduction described its operand with "
                   "a bare data() pointer, so the tracker filed the read under a resource "
                   "nothing else touches and emitted no edge at all.";

            // And the same thing behaviourally, for the reader who wants to see it.
            engine->elementwise().mul(source, source, packed);
            engine->reductions().sum(packed, total);
            engine->execute().wait();

            EXPECT_TENSOR_NEAR(total, {static_cast<float>(kN) * 9.0f}, 1e-5);
        }

    } // namespace Test
} // namespace SushiBLAS
