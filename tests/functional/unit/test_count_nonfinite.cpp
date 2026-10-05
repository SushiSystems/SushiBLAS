/****************************************************************/
/* test_count_nonfinite.cpp                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// `reductions().count_nonfinite` - the exact non-finite screen a loss-scaling
// step runs before it decides whether to apply an update.
//
// Three things this file is deliberately careful about.
//
// **Infinity and NaN are tested separately.** They are two different failures
// with two different tests, and the obvious hand-rolled predicate `x != x`
// catches only one of them - an infinity compares equal to itself. A screen that
// found NaNs and waved infinities through would pass a suite that only ever fed
// it NaNs, and fp16 overflow produces infinities first.
//
// **There is a control.** A clean tensor must count zero. Without it, an
// implementation that returned a positive number unconditionally would pass every
// other case here.
//
// **The last index has its own case.** A fold whose loop bound is off by one
// drops exactly the final element and is otherwise perfect, so the bad element is
// planted at index 0, at the middle, and at n-1 in turn - at a size above
// `Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS`, where the two-stage kernel with its
// grid-strided walk and halving tree is the one running.
//
// The `asum` comparison at the bottom is the load-bearing case for the whole op
// existing. It is a measurement, not an assertion of intent: it runs both ops on
// one tensor and shows them disagreeing.

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/reduce_1d.hpp>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief A size the CPU folds in one sequential pass. */
            constexpr int64_t kSmall = 1024;

            /**
             * @brief A size past the CPU's two-stage threshold.
             *
             * Above `Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS` the reduction takes the
             * grid-strided pass, the local-memory halving tree and the second kernel over
             * the partials - three more places for an element to be dropped than the
             * single-pass shape has.
             */
            constexpr int64_t kLarge = 20000;

            static_assert(kLarge > static_cast<int64_t>(Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS),
                          "kLarge must reach the two-stage shape or half of this file tests "
                          "the same kernel twice.");
        }

        template<typename T>
        class Unit_CountNonFiniteTyped : public SushiBLASTest
        {
            protected:
                /** @brief Positive infinity in the element type under test. */
                static T inf() { return static_cast<T>(std::numeric_limits<float>::infinity()); }

                /** @brief A quiet NaN in the element type under test. */
                static T nan() { return static_cast<T>(std::numeric_limits<float>::quiet_NaN()); }

                /** @brief `n` finite, unremarkable values. */
                static std::vector<T> clean(int64_t n)
                {
                    std::vector<T> data(static_cast<size_t>(n));
                    for (size_t i = 0; i < data.size(); ++i)
                        data[i] = static_cast<T>(static_cast<float>(i % 7) - 3.0f);
                    return data;
                }

                /** @brief Records and runs the op over @p data, returning the count. */
                int64_t count(const std::vector<T>& data)
                {
                    auto t = this->template create_initialized_tensor<T>(
                        {static_cast<int64_t>(data.size())}, data);
                    auto result = this->engine->create_tensor({1}, Core::DataType::INT64);

                    this->engine->reductions().count_nonfinite(t, result);
                    this->engine->execute().wait();

                    int64_t host = -1;
                    TestEnvironment::get_context().get_queue()
                        .memcpy(&host, result.data(), sizeof(int64_t)).wait();
                    return host;
                }
        };

        using CountTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Unit_CountNonFiniteTyped, CountTypes);

        // The control. Every other case below asserts a positive number, and without this
        // one an implementation that always returned a positive number would pass them all.
        TYPED_TEST(Unit_CountNonFiniteTyped, ACleanTensorCountsZero)
        {
            EXPECT_EQ(0, this->count(this->clean(kSmall)));
            EXPECT_EQ(0, this->count(this->clean(kLarge)));
        }

        // Infinity, on its own, in both signs. `x != x` - the reflex NaN test - is false
        // for both of these, so an implementation written that way fails here and nowhere
        // else in this file.
        TYPED_TEST(Unit_CountNonFiniteTyped, AnInfinityIsCountedInEitherSign)
        {
            using T = TypeParam;

            auto positive = this->clean(kSmall);
            positive[17] = this->inf();
            EXPECT_EQ(1, this->count(positive));

            auto negative = this->clean(kSmall);
            negative[17] = static_cast<T>(-this->inf());
            EXPECT_EQ(1, this->count(negative));
        }

        // NaN, on its own. Separate from the infinity case on purpose: a screen built out
        // of a range comparison rather than `isfinite` would catch the infinities and let
        // every NaN through, and only this case would notice.
        TYPED_TEST(Unit_CountNonFiniteTyped, ANaNIsCounted)
        {
            auto data = this->clean(kSmall);
            data[17] = this->nan();
            EXPECT_EQ(1, this->count(data));
        }

        // One bad element in a large tensor, moved to each of the three places a fold can
        // lose it: the first index, somewhere in the middle, and the last - which is where
        // an off-by-one loop bound drops it and nowhere else.
        TYPED_TEST(Unit_CountNonFiniteTyped, ASingleBadElementIsFoundAtAnyIndexIncludingTheLast)
        {
            const std::vector<size_t> where = {0,
                                               static_cast<size_t>(kLarge) / 2,
                                               static_cast<size_t>(kLarge) - 1};

            for (size_t index : where)
            {
                auto with_inf = this->clean(kLarge);
                with_inf[index] = this->inf();
                EXPECT_EQ(1, this->count(with_inf)) << "an infinity at index " << index;

                auto with_nan = this->clean(kLarge);
                with_nan[index] = this->nan();
                EXPECT_EQ(1, this->count(with_nan)) << "a NaN at index " << index;
            }
        }

        // The answer is a count and not a flag: several bad elements of both kinds, spread
        // across the tensor, must add up. This is what distinguishes the op from a
        // saturating "any" and is the extra information the INT64 result buys.
        TYPED_TEST(Unit_CountNonFiniteTyped, EveryBadElementIsCounted)
        {
            using T = TypeParam;

            auto data = this->clean(kLarge);
            data[0] = this->inf();
            data[1] = static_cast<T>(-this->inf());
            data[9999] = this->nan();
            data[static_cast<size_t>(kLarge) - 1] = this->inf();

            EXPECT_EQ(4, this->count(data));
        }

        // The same tensor content at a size either side of the CPU's single-pass/two-stage
        // crossover. Both shapes must give the same count, which is the property that lets
        // the threshold be a pure performance choice.
        TYPED_TEST(Unit_CountNonFiniteTyped, BothLaunchShapesGiveTheSameCount)
        {
            // Not named `small`: <rpcndr.h>, which reaches this translation unit through
            // the Windows headers, defines `small` as a macro for `char`.
            auto below = this->clean(kSmall);
            below[0] = this->nan();
            below[static_cast<size_t>(kSmall) - 1] = this->inf();

            auto above = this->clean(kLarge);
            above[0] = this->nan();
            above[static_cast<size_t>(kLarge) - 1] = this->inf();

            EXPECT_EQ(2, this->count(below)) << "the single-pass shape";
            EXPECT_EQ(2, this->count(above)) << "the two-stage shape";
        }

        // =====================================================================
        // The rest, which is not per-dtype
        // =====================================================================

        class Unit_CountNonFiniteTest : public SushiBLASTest
        {
            protected:
                /** @brief The INT64 scalar @p t holds. */
                int64_t read_count(const Tensor& t)
                {
                    int64_t host = -1;
                    TestEnvironment::get_context().get_queue()
                        .memcpy(&host, t.data(), sizeof(int64_t)).wait();
                    return host;
                }

                /** @brief The HALF scalar @p t holds, widened to float. */
                float read_half(const Tensor& t)
                {
                    sycl::half host{};
                    TestEnvironment::get_context().get_queue()
                        .memcpy(&host, t.data(), sizeof(sycl::half)).wait();
                    return static_cast<float>(host);
                }

                /** @brief The message of the `std::runtime_error` @p fn throws, or "". */
                template<typename Fn>
                std::string message_of(Fn&& fn)
                {
                    try
                    {
                        fn();
                    }
                    catch (const std::runtime_error& e)
                    {
                        return std::string(e.what());
                    }
                    return std::string();
                }
        };

        // The case that decides whether this op needed to exist at all.
        //
        // `asum` would nearly have served - a sum of magnitudes is non-finite whenever any
        // element is - and a caller could have written `isfinite(asum(g))` instead. Here
        // both ops run on ONE tensor of eight perfectly finite fp16 values, and they
        // disagree: the tensor is clean, and `asum` says it is not, because it accumulates
        // in the input's own dtype and 8 x 20000 leaves binary16's range long before any
        // element does.
        //
        // In a loss-scaling loop that disagreement is not a rounding nuisance. It halves
        // the scale on a step that never overflowed, and does it again on the next step,
        // for ever - the mechanism meant to rescue a mixed-precision run would be the thing
        // that stalls it. Nothing here is asserted about which value `asum` returns, only
        // that it is not finite; that is the whole claim.
        TEST_F(Unit_CountNonFiniteTest, AsumReportsOverflowOnAHalfTensorThisOpCallsClean)
        {
            const std::vector<sycl::half> values(8, static_cast<sycl::half>(20000.0f));

            auto t = create_initialized_tensor<sycl::half>({8}, values);
            auto count = engine->create_tensor({1}, Core::DataType::INT64);
            auto total = engine->create_tensor({1}, Core::DataType::HALF);

            engine->reductions().count_nonfinite(t, count);
            engine->reductions().asum(t, total);
            engine->execute().wait();

            EXPECT_EQ(0, read_count(count))
                << "Every element is 20000, which binary16 represents exactly. The tensor "
                   "is clean and the count must say so.";
            EXPECT_FALSE(std::isfinite(read_half(total)))
                << "If this ever becomes finite, re-read the argument on "
                   "ReductionOps::count_nonfinite: it rests on asum overflowing here, and "
                   "if it no longer does then the case for a dedicated op is weaker than "
                   "the header claims.";
        }

        // Integer and complex inputs are refused by name rather than answered. An integer
        // element cannot be infinite or NaN, so "0" would be true and useless - it would
        // tell a caller who handed over the wrong tensor that the right one was screened.
        TEST_F(Unit_CountNonFiniteTest, NonRealInputDTypesAreRefusedByName)
        {
            auto result = engine->create_tensor({1}, Core::DataType::INT64);

            for (Core::DataType dtype : {Core::DataType::INT32, Core::DataType::INT64,
                                         Core::DataType::COMPLEX32, Core::DataType::COMPLEX64})
            {
                auto t = engine->create_tensor({4}, dtype);
                const std::string msg = message_of([&]{ engine->reductions().count_nonfinite(t, result); });
                EXPECT_NE(std::string::npos, msg.find(Core::to_string(dtype)))
                    << "dtype " << Core::to_string(dtype) << " gave: " << msg;
            }

            EXPECT_TRUE(engine->get_graph().take_nodes().empty())
                << "A refused reduction must not leave a task on the graph.";
        }

        TEST_F(Unit_CountNonFiniteTest, TheResultTensorMustBeAnInt64Scalar)
        {
            auto t = engine->create_tensor({4}, Core::DataType::FLOAT32);

            auto wrong_dtype = engine->create_tensor({1}, Core::DataType::FLOAT32);
            const std::string dtype_msg =
                message_of([&]{ engine->reductions().count_nonfinite(t, wrong_dtype); });
            EXPECT_NE(std::string::npos, dtype_msg.find("INT64")) << dtype_msg;

            auto not_scalar = engine->create_tensor({2}, Core::DataType::INT64);
            const std::string scalar_msg =
                message_of([&]{ engine->reductions().count_nonfinite(t, not_scalar); });
            EXPECT_NE(std::string::npos, scalar_msg.find("scalar")) << scalar_msg;
        }

        // The input is a view partway into an arena, for the reason the convert suite spells
        // out: at offset zero a bare `data()` pointer and a real `region()` name the same
        // resource, so the assertion would pass without proving anything. Here they differ.
        TEST_F(Unit_CountNonFiniteTest, IsOrderedBehindTheOpThatWroteItsInput)
        {
            constexpr int64_t kN = 8;
            auto arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());

            Tensor seed(arena, {kN}, Core::DataType::FLOAT32, 0);
            Tensor gradient(arena, {kN}, Core::DataType::FLOAT32, kN);
            auto count = engine->create_tensor({1}, Core::DataType::INT64);

            fill_tensor(seed, std::vector<float>(static_cast<size_t>(kN), 3.0f));

            engine->elementwise().mul(seed, seed, gradient);
            engine->reductions().count_nonfinite(gradient, count);

            auto nodes = engine->get_graph().take_nodes();
            ASSERT_EQ(2u, nodes.size());
            EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                << "The screen reading `gradient` must depend on the op that produced it. "
                   "An in-degree of 0 means the read was declared from a bare pointer, so "
                   "the tracker filed it under a resource nothing else touches - and the "
                   "screen would then be free to run before the gradients arrived, which "
                   "is the most expensive possible way to get a clean answer.";
        }

        // And the same thing behaviourally, end to end: a value that overflows to infinity
        // inside the graph is found by a screen recorded in the same graph.
        TEST_F(Unit_CountNonFiniteTest, AnOverflowProducedInsideTheGraphIsFound)
        {
            auto big = create_initialized_tensor<float>({4}, {1e30f, 1.0f, 1.0f, 1.0f});
            auto squared = engine->create_tensor({4}, Core::DataType::FLOAT32);
            auto count = engine->create_tensor({1}, Core::DataType::INT64);

            // 1e30 * 1e30 overflows binary32; the other three stay at 1.
            engine->elementwise().mul(big, big, squared);
            engine->reductions().count_nonfinite(squared, count);
            engine->execute().wait();

            EXPECT_EQ(1, read_count(count));
        }

    } // namespace Test
} // namespace SushiBLAS
