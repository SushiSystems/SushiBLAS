/**************************************************************************/
/* test_integer_arithmetic.cpp                                            */
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
 * @file test_integer_arithmetic.cpp
 * @brief Tests integer elementwise arithmetic, wrap-around included, and double scalars.
 * @author Mustafa Garip
 */

#include <limits>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /**
             * @brief Holds the hand-worked operands and results for one integer type.
             * @tparam Int int32_t or int64_t.
             */
            template<typename Int>
            struct IntegerCase
            {
                /// @brief The largest value of Int.
                static constexpr Int K_MAX = std::numeric_limits<Int>::max();

                /// @brief The smallest value of Int.
                static constexpr Int K_MIN = std::numeric_limits<Int>::min();

                /// @brief Left operand.
                static std::vector<Int> a() { return {7, -3, K_MAX, K_MIN}; }

                /// @brief Right operand.
                static std::vector<Int> b() { return {5, 4, 1, -1}; }

                /// @brief a + b; MAX + 1 wraps to MIN, MIN + -1 wraps to MAX.
                static std::vector<Int> sum() { return {12, 1, K_MIN, K_MAX}; }

                /// @brief a - b; MIN - -1 is MIN + 1.
                static std::vector<Int> difference() { return {2, -7, K_MAX - 1, K_MIN + 1}; }

                /// @brief a * b; MIN * -1 wraps to MIN.
                static std::vector<Int> product() { return {35, -12, K_MAX, K_MIN}; }

                /// @brief min(a, b).
                static std::vector<Int> minimum() { return {5, -3, 1, K_MIN}; }

                /// @brief max(a, b).
                static std::vector<Int> maximum() { return {7, 4, K_MAX, -1}; }
            };

            /// @brief Lower clamp bound; truncates to -2 on an integer tensor.
            constexpr double K_CLAMP_LOW = -2.5;

            /// @brief Upper clamp bound; truncates to 7 on an integer tensor.
            constexpr double K_CLAMP_HIGH = 7.9;

            /// @brief A FLOAT64 lower bound that rounds to exactly 1.0f in float.
            constexpr double K_FINE_LOW = 1.0 + 1e-9;

            /// @brief An upper bound above every element of the double scalar case.
            constexpr double K_FINE_HIGH = 2.0;
        } // namespace

        /**
         * @brief Runs the integer arithmetic cases for one element type.
         * @tparam Int int32_t or int64_t.
         */
        template<typename Int>
        class Unit_IntegerArithmeticTest : public SushiBLASTest
        {
            protected:
                /// @brief The case table for Int.
                using Case = IntegerCase<Int>;

                /**
                 * @brief Runs a binary op on Case::a() and Case::b() and returns the result.
                 * @param op Callable `(ElementwiseOps&&, const Tensor&, const Tensor&, Tensor&)`.
                 */
                template<typename Op>
                std::vector<Int> run_binary(Op op)
                {
                    auto a = this->template create_initialized_tensor<Int>({4}, Case::a());
                    auto b = this->template create_initialized_tensor<Int>({4}, Case::b());
                    auto c = this->engine->create_tensor({4}, sushiblas_dtype<Int>());

                    op(SushiBLAS::ElementwiseOps(*this->engine), a, b, c);
                    this->engine->execute().value();
                    return read_tensor<Int>(c);
                }
        };

        /// @brief The integer element types under test.
        using IntegerTypes = ::testing::Types<int32_t, int64_t>;
        TYPED_TEST_SUITE(Unit_IntegerArithmeticTest, IntegerTypes, DefaultTypeNames);

        /**
         * @brief Tests that add wraps on overflow.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, AddWrapsAroundOnOverflow)
        {
            const auto got = this->run_binary([](auto&& ew, const Tensor& a, const Tensor& b, Tensor& c)
                                              { ew.add(a, b, c); });
            EXPECT_EQ(TestFixture::Case::sum(), got);
        }

        /**
         * @brief Tests sub on hand values.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, SubtractsElementwise)
        {
            const auto got = this->run_binary([](auto&& ew, const Tensor& a, const Tensor& b, Tensor& c)
                                              { ew.sub(a, b, c); });
            EXPECT_EQ(TestFixture::Case::difference(), got);
        }

        /**
         * @brief Tests that mul wraps when MIN is multiplied by -1.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, MulWrapsAroundOnOverflow)
        {
            const auto got = this->run_binary([](auto&& ew, const Tensor& a, const Tensor& b, Tensor& c)
                                              { ew.mul(a, b, c); });
            EXPECT_EQ(TestFixture::Case::product(), got);
        }

        /**
         * @brief Tests min on hand values.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, MinPicksTheSmallerElement)
        {
            const auto got = this->run_binary([](auto&& ew, const Tensor& a, const Tensor& b, Tensor& c)
                                              { ew.min(a, b, c); });
            EXPECT_EQ(TestFixture::Case::minimum(), got);
        }

        /**
         * @brief Tests max on hand values.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, MaxPicksTheLargerElement)
        {
            const auto got = this->run_binary([](auto&& ew, const Tensor& a, const Tensor& b, Tensor& c)
                                              { ew.max(a, b, c); });
            EXPECT_EQ(TestFixture::Case::maximum(), got);
        }

        /**
         * @brief Tests that neg and abs map MIN to itself and other values as usual.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, NegAndAbsMapTheMinimumToItself)
        {
            using Int = TypeParam;
            constexpr Int k_min = TestFixture::Case::K_MIN;

            auto n = this->template create_initialized_tensor<Int>({3}, {5, -3, k_min});
            auto m = this->template create_initialized_tensor<Int>({3}, {5, -3, k_min});

            SushiBLAS::ElementwiseOps(*this->engine).neg(n);
            SushiBLAS::ElementwiseOps(*this->engine).abs(m);
            this->engine->execute().value();

            EXPECT_EQ((std::vector<Int>{-5, 3, k_min}), read_tensor<Int>(n));
            EXPECT_EQ((std::vector<Int>{5, 3, k_min}), read_tensor<Int>(m));
        }

        /**
         * @brief Tests that clamp truncates its double bounds toward zero on an integer tensor.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, ClampTruncatesItsBoundsTowardZero)
        {
            using Int = TypeParam;
            auto t = this->template create_initialized_tensor<Int>({3}, {-10, 0, 10});

            SushiBLAS::ElementwiseOps(*this->engine).clamp(t, K_CLAMP_LOW, K_CLAMP_HIGH);
            this->engine->execute().value();

            EXPECT_EQ((std::vector<Int>{-2, 0, 7}), read_tensor<Int>(t));
        }

        /**
         * @brief Tests that clamp throws at record time when min_val exceeds max_val.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, ClampRejectsAnInvertedRange)
        {
            GraphProbe probe;

            using Int = TypeParam;
            auto t = this->template create_initialized_tensor<Int>({3}, {-10, 0, 10});

            EXPECT_THROW(
                SushiBLAS::ElementwiseOps(probe.recorder()).clamp(t, K_CLAMP_HIGH, K_CLAMP_LOW),
                std::invalid_argument);
            EXPECT_EQ(0u, probe.take_recorded_count());
        }

        /**
         * @brief Tests that the safe_ew_add functor wraps MAX + 1 to MIN.
         */
        TYPED_TEST(Unit_IntegerArithmeticTest, SafeAddFunctorWrapsAroundOnOverflow)
        {
            using Int = TypeParam;
            using Case = typename TestFixture::Case;
            auto a = this->template create_initialized_tensor<Int>({2}, {Case::K_MAX, 7});
            auto b = this->template create_initialized_tensor<Int>({2}, {1, 5});
            auto c = this->engine->create_tensor({2}, sushiblas_dtype<Int>());

            Dispatch::execute_binary<Dispatch::Domain::RealAndIntegral>(
                this->engine->recorder(), a, b, c, "test.safe_ew_add", "test.safe_ew_add"_op,
                [](auto x, auto y)
                {
                    return Dispatch::safe_ew_add(x, y);
                });
            this->engine->execute().value();

            EXPECT_EQ((std::vector<Int>{Case::K_MIN, 12}), read_tensor<Int>(c));
        }

        /// @brief Runs the double scalar parameter case.
        class Unit_DoubleScalarTest : public SushiBLASTest {};

        /**
         * @brief Tests that clamp keeps a 1e-9 offset on a FLOAT64 tensor that float would drop.
         */
        TEST_F(Unit_DoubleScalarTest, ClampKeepsADoubleBoundOnAFloat64Tensor)
        {
            auto t = create_initialized_tensor<double>({2}, {1.0, 1.5});

            SushiBLAS::ElementwiseOps(*engine).clamp(t, K_FINE_LOW, K_FINE_HIGH);
            engine->execute().value();

            const std::vector<double> got = read_tensor<double>(t);
            EXPECT_DOUBLE_EQ(K_FINE_LOW, got[0]);
            EXPECT_DOUBLE_EQ(1.5, got[1]);
        }
    } // namespace Test
} // namespace SushiBLAS
