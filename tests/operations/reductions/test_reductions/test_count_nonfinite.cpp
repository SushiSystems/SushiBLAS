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
 * @file test_count_nonfinite.cpp
 * @brief Tests ReductionOps::count_nonfinite, its refusals and its ordering in the graph.
 * @author Mustafa Garip
 */

#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

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
        class Medium_CountNonFiniteTyped : public SushiBLASTest
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

                    SushiBLAS::ReductionOps(this->recorder()).count_nonfinite(t, result);
                    this->recording->run().value();

                    int64_t host = -1;
                    TestEnvironment::queue()
                        .memcpy(&host, result.data(), sizeof(int64_t)).wait();
                    return host;
                }

                /** @brief `n` clean values with a NaN first and an infinity last. */
                static std::vector<T> bad_at_both_ends(int64_t n)
                {
                    auto data = clean(n);
                    data.front() = nan();
                    data.back() = inf();
                    return data;
                }

                /** @brief Expects one infinity and then one NaN, planted at @p index of K_LARGE, to count one each. */
                void expect_one_bad_element_found_at(size_t index)
                {
                    auto with_inf = clean(K_LARGE);
                    with_inf[index] = inf();
                    EXPECT_EQ(1, count(with_inf)) << "an infinity at index " << index;

                    auto with_nan = clean(K_LARGE);
                    with_nan[index] = nan();
                    EXPECT_EQ(1, count(with_nan)) << "a NaN at index " << index;
                }
        };

        using CountTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Medium_CountNonFiniteTyped, CountTypes);

        /**
         * @brief Tests that a clean tensor counts zero.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, ACleanTensorCountsZero)
        {
            EXPECT_EQ(0, this->count(this->clean(K_SMALL)));
            EXPECT_EQ(0, this->count(this->clean(K_LARGE)));
        }

        /**
         * @brief Tests that large finite values count zero, however far their sum would overflow.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, LargeFiniteValuesCountZero)
        {
            using T = TypeParam;

            // 20000 is exact in binary16, and eight of them sum past its range.
            EXPECT_EQ(0, this->count(std::vector<T>(8, static_cast<T>(20000.0f))));
        }

        /**
         * @brief Tests that a positive infinity is counted.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, APositiveInfinityIsCounted)
        {
            auto data = this->clean(K_SMALL);
            data[17] = this->inf();
            EXPECT_EQ(1, this->count(data));
        }

        /**
         * @brief Tests that a negative infinity is counted.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, ANegativeInfinityIsCounted)
        {
            using T = TypeParam;

            auto data = this->clean(K_SMALL);
            data[17] = static_cast<T>(-this->inf());
            EXPECT_EQ(1, this->count(data));
        }

        /**
         * @brief Tests that a NaN is counted.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, ANaNIsCounted)
        {
            auto data = this->clean(K_SMALL);
            data[17] = this->nan();
            EXPECT_EQ(1, this->count(data));
        }

        /**
         * @brief Tests that one bad element is found at the first index.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, ASingleBadElementIsFoundAtTheFirstIndex)
        {
            this->expect_one_bad_element_found_at(0);
        }

        /**
         * @brief Tests that one bad element is found at a middle index.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, ASingleBadElementIsFoundAtTheMiddleIndex)
        {
            this->expect_one_bad_element_found_at(static_cast<size_t>(K_LARGE) / 2);
        }

        /**
         * @brief Tests that one bad element is found at the last index, where an off-by-one drops it.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, ASingleBadElementIsFoundAtTheLastIndex)
        {
            this->expect_one_bad_element_found_at(static_cast<size_t>(K_LARGE) - 1);
        }

        /**
         * @brief Tests that several bad elements of both kinds add up to their count.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, EveryBadElementIsCounted)
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
         * @brief Tests that a size below the CPU's two-stage threshold counts both bad ends.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, TheSinglePassShapeCountsBothBadEnds)
        {
            // `small` is a macro for `char` in <rpcndr.h>. See the test README.
            EXPECT_EQ(2, this->count(this->bad_at_both_ends(K_SMALL)));
        }

        /**
         * @brief Tests that a size above the CPU's two-stage threshold counts both bad ends.
         * @see tests/operations/reductions/README.md
         */
        TYPED_TEST(Medium_CountNonFiniteTyped, TheTwoStageShapeCountsBothBadEnds)
        {
            EXPECT_EQ(2, this->count(this->bad_at_both_ends(K_LARGE)));
        }

        /**
         * @brief Returns the message of the `std::invalid_argument` fn throws, or "".
         */
        template<typename Fn>
        static std::string message_of(Fn&& fn)
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

        class Medium_CountNonFiniteTest : public SushiBLASTest
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
        };

        /** @brief Names a dtype row by the dtype's name. */
        static std::string dtype_name(const ::testing::TestParamInfo<Core::DataType>& info)
        {
            return Core::to_string(info.param);
        }

        /** @brief Records count_nonfinite over an input dtype the op refuses. */
        class Medium_CountNonFiniteDtype
            : public SushiBLASTest,
              public ::testing::WithParamInterface<Core::DataType>
        {
        };

        /**
         * @brief Tests that integer and complex inputs are refused by name.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_CountNonFiniteDtype, NonRealInputIsRefusedByName)
        {
            GraphProbe probe;
            auto result = engine->create_tensor({1}, Core::DataType::Int64);
            auto t = engine->create_tensor({4}, GetParam());

            const std::string msg = message_of(
                [&]
                {
                    SushiBLAS::ReductionOps(probe.recorder()).count_nonfinite(t, result);
                });

            EXPECT_NE(std::string::npos, msg.find(Core::to_string(GetParam()))) << msg;
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_CountNonFiniteDtype,
                                 ::testing::Values(Core::DataType::Int32, Core::DataType::Int64,
                                                   Core::DataType::Complex32,
                                                   Core::DataType::Complex64),
                                 dtype_name);

        /**
         * @brief Tests that a refused reduction leaves no task on the graph.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_CountNonFiniteTest, ARefusedReductionLeavesNoTaskOnTheGraph)
        {
            GraphProbe probe;
            auto result = engine->create_tensor({1}, Core::DataType::Int64);
            auto t = engine->create_tensor({4}, Core::DataType::Int32);

            EXPECT_THROW(SushiBLAS::ReductionOps(probe.recorder()).count_nonfinite(t, result),
                         std::invalid_argument);

            EXPECT_EQ(0u, probe.take_recorded_count());
        }

        /**
         * @brief Tests that a result of another dtype than INT64 is refused by name.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_CountNonFiniteTest, TheResultTensorMustBeInt64)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float32);
            auto wrong_dtype = engine->create_tensor({1}, Core::DataType::Float32);

            const std::string msg = message_of(
                [&]{ SushiBLAS::ReductionOps(recorder()).count_nonfinite(t, wrong_dtype); });

            EXPECT_NE(std::string::npos, msg.find("INT64")) << msg;
        }

        /**
         * @brief Tests that an INT64 result that is not a scalar is refused by name.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_CountNonFiniteTest, TheResultTensorMustBeAScalar)
        {
            auto t = engine->create_tensor({4}, Core::DataType::Float32);
            auto not_scalar = engine->create_tensor({2}, Core::DataType::Int64);

            const std::string msg = message_of(
                [&]{ SushiBLAS::ReductionOps(recorder()).count_nonfinite(t, not_scalar); });

            EXPECT_NE(std::string::npos, msg.find("scalar")) << msg;
        }

        /**
         * @brief Tests that the op is ordered behind the op that wrote its offset input view.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_CountNonFiniteTest, IsOrderedBehindTheOpThatWroteItsInput)
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
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_CountNonFiniteTest, AnOverflowProducedInsideTheGraphIsFound)
        {
            auto big = create_initialized_tensor<float>({4}, {1e30f, 1.0f, 1.0f, 1.0f});
            auto squared = engine->create_tensor({4}, Core::DataType::Float32);
            auto count = engine->create_tensor({1}, Core::DataType::Int64);

            // 1e30 * 1e30 overflows binary32; the other three stay at 1.
            SushiBLAS::ElementwiseOps(recorder()).mul(big, big, squared);
            SushiBLAS::ReductionOps(recorder()).count_nonfinite(squared, count);
            recording->run().value();

            EXPECT_EQ(1, read_count(count));
        }

    } // namespace Test
} // namespace SushiBLAS
