/**************************************************************************/
/* test_count_nonfinite.cpp                                               */
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
 * @file test_count_nonfinite.cpp
 * @brief Tests ReductionOps::count_nonfinite, its refusals and its ordering in the graph.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include "test_utils.hpp"
#include "graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief A size the CPU folds in one sequential pass. */
            constexpr int64_t K_SMALL = 1024;

            /**
             * @brief A size past the CPU's two-stage threshold.
             *
             * Above `Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS` the reduction takes the
             * grid-strided pass, the local-memory halving tree and the second kernel over
             * the partials - three more places for an element to be dropped than the
             * single-pass shape has.
             */
            constexpr int64_t K_LARGE = 20000;

            static_assert(K_LARGE > static_cast<int64_t>(Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS),
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
                    auto result = this->engine->create_tensor({1}, Core::DataType::Int64);

                    SushiBLAS::ReductionOps(*this->engine).count_nonfinite(t, result);
                    this->engine->execute().value();

                    int64_t host = -1;
                    TestEnvironment::queue()
                        .memcpy(&host, result.data(), sizeof(int64_t)).wait();
                    return host;
                }
        };

        using CountTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Unit_CountNonFiniteTyped, CountTypes);

        /**
         * @brief Tests that a clean tensor counts zero.
         * @see tests/unit/README.md
         */
        TYPED_TEST(Unit_CountNonFiniteTyped, ACleanTensorCountsZero)
        {
            EXPECT_EQ(0, this->count(this->clean(K_SMALL)));
            EXPECT_EQ(0, this->count(this->clean(K_LARGE)));
        }

        /**
         * @brief Tests that an infinity of either sign is counted.
         * @see tests/unit/README.md
         */
        TYPED_TEST(Unit_CountNonFiniteTyped, AnInfinityIsCountedInEitherSign)
        {
            using T = TypeParam;

            auto positive = this->clean(K_SMALL);
            positive[17] = this->inf();
            EXPECT_EQ(1, this->count(positive));

            auto negative = this->clean(K_SMALL);
            negative[17] = static_cast<T>(-this->inf());
            EXPECT_EQ(1, this->count(negative));
        }

        /**
         * @brief Tests that a NaN is counted.
         * @see tests/unit/README.md
         */
        TYPED_TEST(Unit_CountNonFiniteTyped, ANaNIsCounted)
        {
            auto data = this->clean(K_SMALL);
            data[17] = this->nan();
            EXPECT_EQ(1, this->count(data));
        }

        /**
         * @brief Tests that one bad element is found at the first, a middle and the last index.
         * @see tests/unit/README.md
         */
        TYPED_TEST(Unit_CountNonFiniteTyped, ASingleBadElementIsFoundAtAnyIndexIncludingTheLast)
        {
            const std::vector<size_t> where = {0,
                                               static_cast<size_t>(K_LARGE) / 2,
                                               static_cast<size_t>(K_LARGE) - 1};

            for (size_t index : where)
            {
                auto with_inf = this->clean(K_LARGE);
                with_inf[index] = this->inf();
                EXPECT_EQ(1, this->count(with_inf)) << "an infinity at index " << index;

                auto with_nan = this->clean(K_LARGE);
                with_nan[index] = this->nan();
                EXPECT_EQ(1, this->count(with_nan)) << "a NaN at index " << index;
            }
        }

        /**
         * @brief Tests that several bad elements of both kinds add up to their count.
         * @see tests/unit/README.md
         */
        TYPED_TEST(Unit_CountNonFiniteTyped, EveryBadElementIsCounted)
        {
            using T = TypeParam;

            auto data = this->clean(K_LARGE);
            data[0] = this->inf();
            data[1] = static_cast<T>(-this->inf());
            data[9999] = this->nan();
            data[static_cast<size_t>(K_LARGE) - 1] = this->inf();

            EXPECT_EQ(4, this->count(data));
        }

        /**
         * @brief Tests that sizes on each side of the CPU's two-stage threshold count the same.
         * @see tests/unit/README.md
         */
        TYPED_TEST(Unit_CountNonFiniteTyped, BothLaunchShapesGiveTheSameCount)
        {
            // `small` is a macro for `char` in <rpcndr.h>. See tests/unit/README.md.
            auto below = this->clean(K_SMALL);
            below[0] = this->nan();
            below[static_cast<size_t>(K_SMALL) - 1] = this->inf();

            auto above = this->clean(K_LARGE);
            above[0] = this->nan();
            above[static_cast<size_t>(K_LARGE) - 1] = this->inf();

            EXPECT_EQ(2, this->count(below)) << "the single-pass shape";
            EXPECT_EQ(2, this->count(above)) << "the two-stage shape";
        }

        class Unit_CountNonFiniteTest : public SushiBLASTest
        {
            protected:
                /** @brief The INT64 scalar @p t holds. */
                int64_t read_count(const Tensor& t)
                {
                    int64_t host = -1;
                    TestEnvironment::queue()
                        .memcpy(&host, t.data(), sizeof(int64_t)).wait();
                    return host;
                }

                /** @brief The HALF scalar @p t holds, widened to float. */
                float read_half(const Tensor& t)
                {
                    sycl::half host{};
                    TestEnvironment::queue()
                        .memcpy(&host, t.data(), sizeof(sycl::half)).wait();
                    return static_cast<float>(host);
                }

                /** @brief The message of the `std::invalid_argument` @p fn throws, or "". */
                template<typename Fn>
                std::string message_of(Fn&& fn)
                {
                    try
                    {
                        fn();
                    }
                    catch (const std::invalid_argument& e)
                    {
                        return std::string(e.what());
                    }
                    return std::string();
                }
        };

        /**
         * @brief Tests that asum overflows on eight finite fp16 values this op counts as clean.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_CountNonFiniteTest, AsumReportsOverflowOnAHalfTensorThisOpCallsClean)
        {
            const std::vector<sycl::half> values(8, static_cast<sycl::half>(20000.0f));

            auto t = create_initialized_tensor<sycl::half>({8}, values);
            auto count = engine->create_tensor({1}, Core::DataType::Int64);
            auto total = engine->create_tensor({1}, Core::DataType::Half);

            SushiBLAS::ReductionOps(*engine).count_nonfinite(t, count);
            SushiBLAS::ReductionOps(*engine).asum(t, total);
            engine->execute().value();

            EXPECT_EQ(0, read_count(count))
                << "Every element is 20000, which binary16 represents exactly. The tensor "
                   "is clean and the count must say so.";
            EXPECT_FALSE(std::isfinite(read_half(total)))
                << "If this ever becomes finite, re-read the argument on "
                   "ReductionOps::count_nonfinite: it rests on asum overflowing here, and "
                   "if it no longer does then the case for a dedicated op is weaker than "
                   "the header claims.";
        }

        /**
         * @brief Tests that integer and complex inputs are refused by name.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_CountNonFiniteTest, NonRealInputDTypesAreRefusedByName)
        {
            GraphProbe probe;

            auto result = engine->create_tensor({1}, Core::DataType::Int64);

            for (Core::DataType dtype : {Core::DataType::Int32, Core::DataType::Int64,
                                         Core::DataType::Complex32, Core::DataType::Complex64})
            {
                auto t = engine->create_tensor({4}, dtype);
                const std::string msg = message_of(
                    [&]
                    {
                        SushiBLAS::ReductionOps(probe.recorder()).count_nonfinite(t, result);
                    });
                EXPECT_NE(std::string::npos, msg.find(Core::to_string(dtype)))
                    << "dtype " << Core::to_string(dtype) << " gave: " << msg;
            }

            EXPECT_EQ(0u, probe.take_recorded_count())
                << "A refused reduction must not leave a task on the graph.";
        }

        TEST_F(Unit_CountNonFiniteTest, TheResultTensorMustBeAnInt64Scalar)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float32);

            auto wrong_dtype = engine->create_tensor({1}, Core::DataType::Float32);
            const std::string dtype_msg = message_of(
                [&]{ SushiBLAS::ReductionOps(*engine).count_nonfinite(t, wrong_dtype); });
            EXPECT_NE(std::string::npos, dtype_msg.find("INT64")) << dtype_msg;

            auto not_scalar = engine->create_tensor({2}, Core::DataType::Int64);
            const std::string scalar_msg = message_of(
                [&]{ SushiBLAS::ReductionOps(*engine).count_nonfinite(t, not_scalar); });
            EXPECT_NE(std::string::npos, scalar_msg.find("scalar")) << scalar_msg;
        }

        /**
         * @brief Tests that the op is ordered behind the op that wrote its offset input view.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_CountNonFiniteTest, IsOrderedBehindTheOpThatWroteItsInput)
        {
            GraphProbe probe;

            constexpr int64_t kN = 8;
            auto arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Residency::Shared);

            Tensor seed(arena, {kN}, Core::DataType::Float32, 0);
            Tensor gradient(arena, {kN}, Core::DataType::Float32, kN);
            auto count = engine->create_tensor({1}, Core::DataType::Int64);

            fill_tensor(seed, std::vector<float>(static_cast<size_t>(kN), 3.0f));

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(seed, seed, gradient);
            SushiBLAS::ReductionOps(probe.recorder()).count_nonfinite(gradient, count);

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks.has_value()) << tasks.error().message;
            EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                << "The screen reading `gradient` must depend on the op that produced it. "
                   "An in-degree of 0 means the read was declared from a bare pointer, so "
                   "the tracker filed it under a resource nothing else touches - and the "
                   "screen would then be free to run before the gradients arrived, which "
                   "is the most expensive possible way to get a clean answer.";
        }

        /**
         * @brief Tests that a screen finds an overflow produced earlier in the same graph.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_CountNonFiniteTest, AnOverflowProducedInsideTheGraphIsFound)
        {
            auto big = create_initialized_tensor<float>({4}, {1e30f, 1.0f, 1.0f, 1.0f});
            auto squared = engine->create_tensor({4}, Core::DataType::Float32);
            auto count = engine->create_tensor({1}, Core::DataType::Int64);

            // 1e30 * 1e30 overflows binary32; the other three stay at 1.
            SushiBLAS::ElementwiseOps(*engine).mul(big, big, squared);
            SushiBLAS::ReductionOps(*engine).count_nonfinite(squared, count);
            engine->execute().value();

            EXPECT_EQ(1, read_count(count));
        }

    } // namespace Test
} // namespace SushiBLAS
