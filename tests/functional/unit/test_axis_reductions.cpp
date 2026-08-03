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

// The segmented reduction seam: `max_along` / `sum_along`, the geometry they are
// planned from, and the two kernels that geometry chooses between.
//
// Every expectation here is worked out by hand rather than taken from another
// path in this library. That is deliberate: a differential oracle against
// `gemv(X, ones)` would have agreed with a row sum that read the wrong stride in
// exactly the layouts where both read it wrongly, which is how a column-major
// GEMM defect survived months in this repository.
//
// Determinism lives here too rather than in test_reduction_determinism.cpp,
// because that file's helpers are built around a scalar result and this seam's
// answer is a vector. The property asserted is the same one, by the same means:
// std::memcmp, not a tolerance.

#include <cmath>
#include <limits>
#include <cstring>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/reduce_axis.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /**
             * @brief A 3x4 matrix whose every row and column reduction is an integer.
             *
             * Small whole numbers on purpose: they are exact in HALF as well as in the
             * wider types, so one expectation serves every dtype and a mismatch is a
             * real disagreement rather than a rounding difference.
             *
             *      1  -2   3   0      row max  3    row sum   2
             *      5   4  -1   2      row max  5    row sum  10
             *     -3  -7  -2  -9      row max -2    row sum -21
             *
             *   col max  5   4   3   2
             *   col sum  3  -5   0  -7
             */
            const std::vector<float> kRowMajor =
                {1, -2, 3, 0,
                 5, 4, -1, 2,
                 -3, -7, -2, -9};

            /** @brief The same logical matrix laid out column by column. */
            const std::vector<float> kColumnMajor =
                {1, 5, -3,
                 -2, 4, -7,
                 3, -1, -2,
                 0, 2, -9};

            const std::vector<float> kRowMax = {3, 5, -2};
            const std::vector<float> kColMax = {5, 4, 3, 2};
            const std::vector<float> kRowSum = {2, 10, -21};
            const std::vector<float> kColSum = {3, -5, 0, -7};

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
        }

        // ---------------------------------------------------------------------
        // The planner, which is a pure function and needs no device at all.
        // ---------------------------------------------------------------------

        class Unit_AxisReductionPlan : public ::testing::Test {};

        // The cheap shape whenever there is already enough work to go round: no
        // local memory, no barriers, and a fold order that is trivially fixed.
        TEST_F(Unit_AxisReductionPlan, EnoughSegmentsTakesOneItemPerSegment)
        {
            const Device::Profile cpu = profile(true, 12, 8192, 8);
            const Device::Profile gpu = profile(false, 108, 1024, 32);

            EXPECT_FALSE(Reduce::plan_axis(cpu, 12, 1 << 20).cooperative());
            EXPECT_FALSE(Reduce::plan_axis(cpu, 4096, 1 << 20).cooperative());

            // A GPU wants K_WAVES_PER_COMPUTE_UNIT groups per unit before it is full.
            EXPECT_FALSE(Reduce::plan_axis(gpu, 108 * Device::K_WAVES_PER_COMPUTE_UNIT,
                                           1 << 20).cooperative());
        }

        // Too few segments to fill the machine, and segments long enough to be worth
        // a work-group: the only case where the tree earns its barriers.
        TEST_F(Unit_AxisReductionPlan, FewLongSegmentsTakeAWorkGroupEach)
        {
            const Device::Profile cpu = profile(true, 12, 8192, 8);
            const Device::Profile gpu = profile(false, 108, 1024, 32);

            const Reduce::AxisPlan cpu_plan = Reduce::plan_axis(cpu, 2, 1 << 20);
            EXPECT_TRUE(cpu_plan.cooperative());
            EXPECT_EQ(Reduce::group_width(cpu), cpu_plan.local);

            const Reduce::AxisPlan gpu_plan = Reduce::plan_axis(gpu, 2, 1 << 20);
            EXPECT_TRUE(gpu_plan.cooperative());
            EXPECT_EQ(Reduce::group_width(gpu), gpu_plan.local);
        }

        // A group whose segment is shorter than the group is mostly idle lanes paying
        // log2(local) pairs of barriers, so the tree is declined below one full pass.
        TEST_F(Unit_AxisReductionPlan, FewShortSegmentsStillTakeOneItemPerSegment)
        {
            const Device::Profile cpu = profile(true, 12, 8192, 8);
            const std::size_t local = Reduce::group_width(cpu);

            EXPECT_FALSE(Reduce::plan_axis(cpu, 2, local - 1).cooperative());
            EXPECT_TRUE(Reduce::plan_axis(cpu, 2, local).cooperative());
        }

        // Degenerate spans must not select a shape that would launch `segments * local`
        // work-items over nothing, or divide by a zero extent.
        TEST_F(Unit_AxisReductionPlan, EmptySpansTakeTheCheapShape)
        {
            const Device::Profile cpu = profile(true, 12, 8192, 8);

            EXPECT_FALSE(Reduce::plan_axis(cpu, 0, 1024).cooperative());
            EXPECT_FALSE(Reduce::plan_axis(cpu, 4, 0).cooperative());
        }

        // ---------------------------------------------------------------------
        // The public operations, over every dtype a reduction serves.
        // ---------------------------------------------------------------------

        template<typename T>
        class Unit_AxisReductionsTyped : public SushiBLASTest {};

        using AxisTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Unit_AxisReductionsTyped, AxisTypes);

        TYPED_TEST(Unit_AxisReductionsTyped, RowMaxFoldsAxisOne)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, get_sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({3}, get_sushiblas_dtype<T>());
            fill_tensor(x, kRowMajor);

            this->engine->reductions().max_along(x, 1, out);
            this->engine->execute().wait();

            verify_tensor(out, kRowMax);
        }

        TYPED_TEST(Unit_AxisReductionsTyped, ColumnMaxFoldsAxisZero)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, get_sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({4}, get_sushiblas_dtype<T>());
            fill_tensor(x, kRowMajor);

            this->engine->reductions().max_along(x, 0, out);
            this->engine->execute().wait();

            verify_tensor(out, kColMax);
        }

        TYPED_TEST(Unit_AxisReductionsTyped, RowSumFoldsAxisOne)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, get_sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({3}, get_sushiblas_dtype<T>());
            fill_tensor(x, kRowMajor);

            this->engine->reductions().sum_along(x, 1, out);
            this->engine->execute().wait();

            verify_tensor(out, kRowSum);
        }

        TYPED_TEST(Unit_AxisReductionsTyped, ColumnSumFoldsAxisZero)
        {
            using T = TypeParam;
            auto x = this->engine->create_tensor({3, 4}, get_sushiblas_dtype<T>());
            auto out = this->engine->create_tensor({4}, get_sushiblas_dtype<T>());
            fill_tensor(x, kRowMajor);

            this->engine->reductions().sum_along(x, 0, out);
            this->engine->execute().wait();

            verify_tensor(out, kColSum);
        }

        // ---------------------------------------------------------------------
        // Layout, shape and rejection.
        // ---------------------------------------------------------------------

        class Unit_AxisReductions : public SushiBLASTest {};

        // The axis is a question about the *matrix*, not about memory order, so the
        // same logical matrix must reduce to the same vectors under either layout.
        // The kernel never learns which layout it is on: `Tensor::strides` already
        // carries it, and `axis_span_for` turns the pair into a stride pair.
        TEST_F(Unit_AxisReductions, ColumnMajorOperandsFoldTheSameAxes)
        {
            reinit_engine(Core::Layout::COLUMN_MAJOR);

            auto x = engine->create_tensor({3, 4});
            auto rows = engine->create_tensor({3});
            auto cols = engine->create_tensor({4});
            fill_tensor(x, kColumnMajor);

            engine->reductions().max_along(x, 1, rows);
            engine->reductions().sum_along(x, 0, cols);
            engine->execute().wait();

            verify_tensor(rows, kRowMax);
            verify_tensor(cols, kColSum);
        }

        // A view at a non-zero storage offset must reduce *its own* elements. Every
        // address the kernel forms is `data() + s * segment_stride + k * element_stride`
        // and `data()` already carries the offset, so this is really a check that the
        // offset was not dropped on the way through the seam.
        TEST_F(Unit_AxisReductions, AViewAtANonZeroOffsetReducesItsOwnElements)
        {
            constexpr int64_t kElements = 12;

            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kElements) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());

            Tensor head(storage, {3, 4}, Core::DataType::FLOAT32, 0);
            Tensor tail(storage, {3, 4}, Core::DataType::FLOAT32, kElements);

            fill_tensor(head, std::vector<float>(static_cast<size_t>(kElements), -1.0f));
            fill_tensor(tail, kRowMajor);

            auto out = engine->create_tensor({3});
            engine->reductions().max_along(tail, 1, out);
            engine->execute().wait();

            verify_tensor(out, kRowMax);
        }

        // The bare-pointer hazard, on the new seam. A region built from `data()`
        // widens to the whole allocation of whatever it points at, so two views of
        // one arena look like two unrelated resources and the tracker emits no edge
        // at all - not a wrong edge, none. That is what cost SushiAI's first
        // arena-packed training run, and `Regression_Sentinel.ReductionsOrder
        // AgainstViewsOfOneAllocation` pins it for the whole-tensor seam.
        TEST_F(Unit_AxisReductions, AxisReductionsOrderAgainstViewsOfOneAllocation)
        {
            constexpr int64_t kElements = 12;

            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kElements) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());

            Tensor source(storage, {3, 4}, Core::DataType::FLOAT32, 0);
            Tensor packed(storage, {3, 4}, Core::DataType::FLOAT32, kElements);
            auto out = engine->create_tensor({3});

            fill_tensor(source, std::vector<float>(static_cast<size_t>(kElements), 3.0f));

            engine->elementwise().mul(source, source, packed);
            engine->reductions().sum_along(packed, 1, out);

            auto nodes = engine->get_graph().take_nodes();
            ASSERT_EQ(2u, nodes.size());
            EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                << "The axis reduction reading `packed` must depend on the op that "
                   "wrote it. An in-degree of 0 means the seam described its operand "
                   "with a bare data() pointer instead of Tensor::region().";

            engine->elementwise().mul(source, source, packed);
            engine->reductions().sum_along(packed, 1, out);
            engine->execute().wait();

            verify_tensor(out, {36.0f, 36.0f, 36.0f});
        }

        TEST_F(Unit_AxisReductions, MalformedOperandsAreRejectedAtRecordTime)
        {
            auto matrix = engine->create_tensor({3, 4});
            auto vector = engine->create_tensor({4});
            auto rows = engine->create_tensor({3});
            auto cols = engine->create_tensor({4});

            // Rank 1 in: there is no axis to fold away.
            EXPECT_THROW(engine->reductions().max_along(vector, 0, rows), std::runtime_error);

            // Rank 3 would be a meaningful request; it is refused by name rather than
            // reduced along the wrong stride.
            auto cube = engine->create_tensor({2, 3, 4});
            EXPECT_THROW(engine->reductions().sum_along(cube, 1, rows), std::runtime_error);

            // Axis out of range.
            EXPECT_THROW(engine->reductions().max_along(matrix, 2, rows), std::runtime_error);
            EXPECT_THROW(engine->reductions().max_along(matrix, -1, rows), std::runtime_error);

            // Result of the wrong length: folding axis 1 of a [3, 4] leaves 3 values.
            EXPECT_THROW(engine->reductions().max_along(matrix, 1, cols), std::runtime_error);
            EXPECT_THROW(engine->reductions().max_along(matrix, 0, rows), std::runtime_error);

            // Result of the wrong rank.
            auto square = engine->create_tensor({3, 1});
            EXPECT_THROW(engine->reductions().sum_along(matrix, 1, square), std::runtime_error);

            // Mismatched dtypes.
            auto halves = engine->create_tensor({3}, Core::DataType::HALF);
            EXPECT_THROW(engine->reductions().sum_along(matrix, 1, halves), std::runtime_error);
        }

        // A fold needs an ordering and an identity, which std::complex does not have,
        // and an integer tensor is not an arithmetic type in this library - the same
        // Domain::REAL rule the whole-tensor reductions carry.
        TEST_F(Unit_AxisReductions, NonRealDtypesAreRefusedByName)
        {
            for (Core::DataType dtype : {Core::DataType::COMPLEX32, Core::DataType::COMPLEX64,
                                         Core::DataType::INT32, Core::DataType::INT64})
            {
                SCOPED_TRACE(Core::to_string(dtype));

                auto x = engine->create_tensor({3, 4}, dtype);
                auto out = engine->create_tensor({3}, dtype);

                EXPECT_THROW(engine->reductions().max_along(x, 1, out), std::runtime_error);
                EXPECT_THROW(engine->reductions().sum_along(x, 1, out), std::runtime_error);
            }
        }

        // ---------------------------------------------------------------------
        // The two kernels, against each other and against exact arithmetic.
        // ---------------------------------------------------------------------

        namespace
        {
            /** @brief Reads a whole tensor back to the host as floats. */
            std::vector<float> host_floats(const Tensor& t)
            {
                std::vector<float> out(static_cast<std::size_t>(t.num_elements));
                TestEnvironment::get_context().get_queue()
                    .memcpy(out.data(), t.data(), out.size() * sizeof(float)).wait();
                return out;
            }
        }

        // `plan_axis` picks one shape per input, so on any single machine the public
        // ops only ever reach one of the two kernels for a given size. This drives
        // both directly, over one input, at a size the planner would never send to
        // the cooperative one - which is the only way the work-group tree gets
        // exercised on a machine with no GPU.
        //
        // The extent is deliberately not a multiple of any work-group width, so the
        // tree's trailing lanes fold fewer elements than the leading ones and the
        // lanes past the extent fold none at all and contribute the identity.
        TEST_F(Unit_AxisReductions, TheCooperativeTreeAgreesWithTheItemPerSegmentFold)
        {
            constexpr std::size_t kSegments = 5;
            constexpr std::size_t kExtent = 733;

            auto& q = TestEnvironment::get_context().get_queue();

            auto in = engine->create_tensor({static_cast<int64_t>(kSegments),
                                             static_cast<int64_t>(kExtent)});
            auto flat = engine->create_tensor({static_cast<int64_t>(kSegments)});
            auto tree = engine->create_tensor({static_cast<int64_t>(kSegments)});

            std::vector<float> host(kSegments * kExtent);
            for (std::size_t s = 0; s < kSegments; ++s)
                for (std::size_t k = 0; k < kExtent; ++k)
                    host[s * kExtent + k] =
                        0.1f + static_cast<float>((s * 7 + k) % 31) * 0.031f;
            q.memcpy(in.data(), host.data(), host.size() * sizeof(float)).wait();

            Reduce::AxisSpan span;
            span.segments = kSegments;
            span.extent = kExtent;
            span.segment_stride = kExtent;
            span.element_stride = 1;
            span.out_stride = 1;

            const auto identity = [](float x) { return x; };
            const auto keep = [](float v, [[maybe_unused]] std::size_t n) { return v; };

            const std::size_t local = Reduce::group_width(Device::profile_for(q));
            ASSERT_GT(local, 1u) << "the cooperative shape needs a group wider than one";

            Reduce::reduce_axis<Reduce::CompensatedSum>(
                q, in.data_as<float>(), flat.data_as<float>(), span,
                Reduce::AxisPlan{1}, identity, keep, {}).wait();

            Reduce::reduce_axis<Reduce::CompensatedSum>(
                q, in.data_as<float>(), tree.data_as<float>(), span,
                Reduce::AxisPlan{local}, identity, keep, {}).wait();

            const std::vector<float> got_flat = host_floats(flat);
            const std::vector<float> got_tree = host_floats(tree);

            for (std::size_t s = 0; s < kSegments; ++s)
            {
                double exact = 0.0;
                for (std::size_t k = 0; k < kExtent; ++k)
                    exact += static_cast<double>(host[s * kExtent + k]);

                // Not bit-equal to each other - they add in different orders, which is
                // the point of having two shapes - but both must be right, and both
                // must be stable run to run, which the case below asserts separately.
                EXPECT_NEAR(static_cast<double>(got_flat[s]), exact, exact * 1e-6)
                    << "item-per-segment, segment " << s;
                EXPECT_NEAR(static_cast<double>(got_tree[s]), exact, exact * 1e-6)
                    << "work-group tree, segment " << s;
            }
        }

        // The same maximum, at both shapes, on data whose per-segment answers differ
        // from the whole-tensor answer by a lot. `max` is exactly associative, so the
        // two shapes must agree to the bit here rather than merely to a tolerance.
        TEST_F(Unit_AxisReductions, BothShapesFindTheSamePerSegmentMaximum)
        {
            constexpr std::size_t kSegments = 5;
            constexpr std::size_t kExtent = 733;

            auto& q = TestEnvironment::get_context().get_queue();

            auto in = engine->create_tensor({static_cast<int64_t>(kSegments),
                                             static_cast<int64_t>(kExtent)});
            auto flat = engine->create_tensor({static_cast<int64_t>(kSegments)});
            auto tree = engine->create_tensor({static_cast<int64_t>(kSegments)});

            // Segment s peaks at 100 * s + 1, at a position that moves with s so no
            // shape can be right by reading a fixed lane.
            std::vector<float> host(kSegments * kExtent, 0.0f);
            std::vector<float> expected(kSegments);
            for (std::size_t s = 0; s < kSegments; ++s)
            {
                for (std::size_t k = 0; k < kExtent; ++k)
                    host[s * kExtent + k] = -static_cast<float>(k % 17);
                host[s * kExtent + (s * 131 + 7) % kExtent] =
                    100.0f * static_cast<float>(s) + 1.0f;
                expected[s] = 100.0f * static_cast<float>(s) + 1.0f;
            }
            q.memcpy(in.data(), host.data(), host.size() * sizeof(float)).wait();

            Reduce::AxisSpan span;
            span.segments = kSegments;
            span.extent = kExtent;
            span.segment_stride = kExtent;
            span.element_stride = 1;
            span.out_stride = 1;

            const auto identity = [](float x) { return x; };
            const auto keep = [](float v, [[maybe_unused]] std::size_t n) { return v; };
            const std::size_t local = Reduce::group_width(Device::profile_for(q));

            Reduce::reduce_axis<Reduce::RunningMax>(
                q, in.data_as<float>(), flat.data_as<float>(), span,
                Reduce::AxisPlan{1}, identity, keep, {}).wait();

            Reduce::reduce_axis<Reduce::RunningMax>(
                q, in.data_as<float>(), tree.data_as<float>(), span,
                Reduce::AxisPlan{local}, identity, keep, {}).wait();

            const std::vector<float> got_flat = host_floats(flat);
            const std::vector<float> got_tree = host_floats(tree);

            for (std::size_t s = 0; s < kSegments; ++s)
            {
                EXPECT_EQ(expected[s], got_flat[s]) << "item-per-segment, segment " << s;
                EXPECT_EQ(expected[s], got_tree[s]) << "work-group tree, segment " << s;
            }
        }

        // Neumaier compensation, per segment. Each row is one large head followed by
        // 4096 ones; in float, 1e8 has an ulp of 8, so `1e8 + 1` rounds straight back
        // to 1e8 and an uncompensated serial fold returns exactly 1e8 - short by the
        // entire tail. 100004096 is a multiple of 8 and therefore exactly
        // representable, so this can assert a value rather than a tolerance.
        TEST_F(Unit_AxisReductions, RowSumsAreCompensated)
        {
            constexpr int64_t kRows = 16;
            constexpr int64_t kOnes = 4096;
            constexpr float kHead = 1e8f;

            auto x = engine->create_tensor({kRows, kOnes + 1});
            std::vector<float> host(static_cast<std::size_t>(kRows * (kOnes + 1)), 1.0f);
            for (int64_t r = 0; r < kRows; ++r)
                host[static_cast<std::size_t>(r * (kOnes + 1))] = kHead;
            TestEnvironment::get_context().get_queue()
                .memcpy(x.data(), host.data(), host.size() * sizeof(float)).wait();

            auto out = engine->create_tensor({kRows});
            engine->reductions().sum_along(x, 1, out);
            engine->execute().wait();

            const std::vector<float> got = host_floats(out);
            for (int64_t r = 0; r < kRows; ++r)
            {
                EXPECT_FLOAT_EQ(kHead + static_cast<float>(kOnes),
                                got[static_cast<std::size_t>(r)])
                    << "row " << r << ": an uncompensated fold returns " << kHead
                    << " here, because every 1.0 falls below the ulp of the running "
                       "total. Neumaier compensation is what recovers the tail.";
            }
        }

        // The guarantee SushiBLAS inherits from `sushiruntime_deterministic_fp` is
        // byte equality for the same binary on the same device, so this compares
        // bytes. A tolerance-based check cannot tell that apart from "close enough",
        // which is the whole distinction under test.
        //
        // Run at both shapes by driving the kernel directly, for the same reason
        // TheCooperativeTreeAgreesWithTheItemPerSegmentFold does.
        TEST_F(Unit_AxisReductions, BothShapesAreBitwiseStable)
        {
            constexpr std::size_t kSegments = 5;
            constexpr std::size_t kExtent = 733;
            constexpr int kRuns = 16;

            auto& q = TestEnvironment::get_context().get_queue();
            const std::size_t local = Reduce::group_width(Device::profile_for(q));

            std::vector<float> host(kSegments * kExtent);
            for (std::size_t i = 0; i < host.size(); ++i)
                host[i] = 0.1f + static_cast<float>(i % 97) * 0.017f;

            Reduce::AxisSpan span;
            span.segments = kSegments;
            span.extent = kExtent;
            span.segment_stride = kExtent;
            span.element_stride = 1;
            span.out_stride = 1;

            const auto identity = [](float x) { return x; };
            const auto keep = [](float v, [[maybe_unused]] std::size_t n) { return v; };

            for (std::size_t width : {std::size_t{1}, local})
            {
                SCOPED_TRACE(width == 1 ? "item-per-segment shape" : "work-group tree");

                std::vector<float> first;
                for (int r = 0; r < kRuns; ++r)
                {
                    Engine eng(TestEnvironment::get_context());
                    auto in = eng.create_tensor({static_cast<int64_t>(kSegments),
                                                 static_cast<int64_t>(kExtent)});
                    auto out = eng.create_tensor({static_cast<int64_t>(kSegments)});
                    q.memcpy(in.data(), host.data(), host.size() * sizeof(float)).wait();

                    Reduce::reduce_axis<Reduce::CompensatedSum>(
                        q, in.data_as<float>(), out.data_as<float>(), span,
                        Reduce::AxisPlan{width}, identity, keep, {}).wait();

                    const std::vector<float> got = host_floats(out);
                    if (r == 0)
                    {
                        // A degenerate result would make byte equality trivial and
                        // this case vacuous, so the input has to reduce to a number
                        // first.
                        for (float v : got)
                            ASSERT_TRUE(std::isfinite(v) && v != 0.0f)
                                << "a segment reduced to " << v << ", which is "
                                   "bit-stable for reasons unrelated to fold order.";
                        first = got;
                        continue;
                    }

                    EXPECT_EQ(0, std::memcmp(first.data(), got.data(),
                                             got.size() * sizeof(float)))
                        << "run " << r << " differs from run 0. A reduction must be "
                           "reproducible to the bit, not merely to a tolerance.";
                }
            }
        }

        // ---------------------------------------------------------------------
        // Why the seam exists.
        // ---------------------------------------------------------------------

        // The numerical property the whole axis reduction was added for, at the
        // SushiBLAS layer: a log-sum-exp shifted by each row's OWN maximum survives a
        // row-to-batch-maximum gap that destroys the globally shifted form.
        //
        // Both spellings are computed here from the same logits, so the comparison is
        // not an appeal to a remembered number. Every row is three equal values, whose
        // log-softmax is exactly -ln(3) for any offset whatsoever - which is what makes
        // the expectation hand-computable and independent of the offset under test.
        TEST_F(Unit_AxisReductions, ARowWiseShiftSurvivesAGapThatDestroysAGlobalOne)
        {
            constexpr int64_t kRows = 2;
            constexpr int64_t kCols = 3;
            constexpr float kGap = 104.0f;

            auto& q = TestEnvironment::get_context().get_queue();

            auto logits = engine->create_tensor({kRows, kCols});
            fill_tensor(logits, {0.0f, 0.0f, 0.0f, -kGap, -kGap, -kGap});

            auto row_max = engine->create_tensor({kRows});
            auto global_max = engine->create_tensor({1});

            engine->reductions().max_along(logits, 1, row_max);
            engine->reductions().max(logits, global_max);
            engine->execute().wait();

            const std::vector<float> peaks = host_floats(row_max);
            ASSERT_EQ(2u, peaks.size());
            EXPECT_FLOAT_EQ(0.0f, peaks[0]);
            EXPECT_FLOAT_EQ(-kGap, peaks[1])
                << "the second row's own maximum is -" << kGap
                << "; a global maximum would report 0 for it.";

            float peak = 0.0f;
            q.memcpy(&peak, global_max.data(), sizeof(float)).wait();
            EXPECT_FLOAT_EQ(0.0f, peak);

            // The row sums of exp(x - shift), under each shift, computed with the
            // same seam that computed the shifts.
            auto shifted = engine->create_tensor({kRows, kCols});
            auto ones_cols = engine->create_tensor({kCols});
            fill_tensor(ones_cols, {1.0f, 1.0f, 1.0f});

            auto row_sums = engine->create_tensor({kRows});

            engine->blas().copy(logits, shifted);
            engine->blas().ger(row_max, ones_cols, shifted, -1.0f);
            engine->elementwise().exp(shifted);
            engine->reductions().sum_along(shifted, 1, row_sums);
            engine->execute().wait();

            const std::vector<float> rowwise = host_floats(row_sums);
            for (std::size_t r = 0; r < rowwise.size(); ++r)
            {
                // exp(0) three times.
                EXPECT_FLOAT_EQ(3.0f, rowwise[r]) << "row " << r;
                EXPECT_TRUE(std::isfinite(std::log(rowwise[r])));
            }

            // And now the same thing shifted by the global maximum, which is what the
            // library offered before this seam existed.
            auto ones_rows = engine->create_tensor({kRows});
            auto peak_row = engine->create_tensor({kCols});
            auto ones_column = engine->create_tensor({kCols, 1});
            fill_tensor(ones_rows, {1.0f, 1.0f});
            fill_tensor(ones_column, {1.0f, 1.0f, 1.0f});

            auto global_sums = engine->create_tensor({kRows});

            engine->blas().gemv(ones_column, global_max, peak_row, false, 1.0f, 0.0f);
            engine->blas().copy(logits, shifted);
            engine->blas().ger(ones_rows, peak_row, shifted, -1.0f);
            engine->elementwise().exp(shifted);
            engine->reductions().sum_along(shifted, 1, global_sums);
            engine->execute().wait();

            const std::vector<float> global = host_floats(global_sums);
            EXPECT_FLOAT_EQ(3.0f, global[0]);
            EXPECT_FLOAT_EQ(0.0f, global[1])
                << "exp(-" << kGap << ") underflows to zero in binary32, so the second "
                   "row's sum of exponentials is exactly zero and its log-softmax is "
                   "+inf. That is the defect the row-wise shift removes.";
            EXPECT_TRUE(std::isinf(std::log(global[1])));
        }

    } // namespace Test
} // namespace SushiBLAS
