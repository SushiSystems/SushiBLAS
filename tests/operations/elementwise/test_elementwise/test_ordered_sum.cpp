/**************************************************************************/
/* test_ordered_sum.cpp                                                   */
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
 * @file test_ordered_sum.cpp
 * @brief Tests ElementwiseOps::ordered_sum against a serial sum of the same order and type, bit for
 *        bit, and its refusals.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Rows of every tensor; the 3 x 347 shape holds 1041 elements, a multiple of no
            ///        work-group size.
            constexpr int64_t K_ROWS = 3;

            /// @brief Columns of every tensor; 347 is prime.
            constexpr int64_t K_COLUMNS = 347;

            /// @brief The element count of every tensor.
            constexpr std::size_t K_ELEMENTS = static_cast<std::size_t>(K_ROWS * K_COLUMNS);

            /// @brief Weights that no float type holds exactly, so each product rounds.
            constexpr std::array<double, 5> K_WEIGHTS = {0.37, -1.9, 2.5, 0.011, -0.73};

            /// @brief Seed of the value generator; every run draws the same values.
            constexpr std::uint64_t K_SEED = 0x5EEDu;

            /// @brief Advances a linear congruential generator and returns its high bits.
            std::uint64_t next_random(std::uint64_t& state)
            {
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                return state >> 11;
            }

            /// @brief Draws a value of about 100 in magnitude with two decimal digits.
            double next_real(std::uint64_t& state)
            {
                return static_cast<double>(static_cast<std::int64_t>(next_random(state) % 20001u) -
                                           10000) /
                       97.0;
            }

            /**
             * @brief Draws K_ELEMENTS values of T from the generator.
             * @tparam T One of the seven element types.
             */
            template<typename T>
            std::vector<T> draw_values(std::uint64_t& state)
            {
                std::vector<T> values(K_ELEMENTS);
                for (T& value : values)
                {
                    if constexpr (std::is_integral_v<T>)
                        value = static_cast<T>(next_random(state) << 20);
                    else if constexpr (std::is_same_v<T, std::complex<float>> ||
                                       std::is_same_v<T, std::complex<double>>)
                    {
                        const double real = next_real(state);
                        const double imaginary = next_real(state);
                        value = T(static_cast<typename T::value_type>(real),
                                  static_cast<typename T::value_type>(imaginary));
                    }
                    else
                        value = static_cast<T>(next_real(state));
                }
                return values;
            }

            /**
             * @brief Sums the inputs element by element in index order, in the accumulation type
             *        the operation names.
             * @tparam T One of the seven element types.
             */
            template <typename T>
            std::vector<T> serial_sum(const std::vector<std::vector<T>>& inputs,
                                      const std::vector<double>& weights)
            {
                // Contraction is off so a multiply and an add round apart, as in the operation.
                #pragma clang fp contract(off)
                std::vector<T> sums(K_ELEMENTS);
                for (std::size_t i = 0; i < K_ELEMENTS; ++i)
                {
                    if constexpr (std::is_integral_v<T>)
                    {
                        using Unsigned = std::make_unsigned_t<T>;
                        Unsigned sum = 0;
                        for (const std::vector<T>& input : inputs)
                            sum = static_cast<Unsigned>(sum + static_cast<Unsigned>(input[i]));
                        sums[i] = static_cast<T>(sum);
                    }
                    else if constexpr (std::is_same_v<T, std::complex<float>> ||
                                       std::is_same_v<T, std::complex<double>>)
                    {
                        using Real = typename T::value_type;
                        Real real_sum = 0;
                        Real imaginary_sum = 0;
                        for (std::size_t k = 0; k < inputs.size(); ++k)
                        {
                            real_sum += static_cast<Real>(weights[k]) * inputs[k][i].real();
                            imaginary_sum += static_cast<Real>(weights[k]) * inputs[k][i].imag();
                        }
                        sums[i] = T(real_sum, imaginary_sum);
                    }
                    else
                    {
                        using Accumulation =
                            std::conditional_t<std::is_same_v<T, sycl::half>, float, T>;
                        Accumulation sum = 0;
                        for (std::size_t k = 0; k < inputs.size(); ++k)
                            sum += static_cast<Accumulation>(weights[k]) *
                                   static_cast<Accumulation>(inputs[k][i]);
                        sums[i] = static_cast<T>(sum);
                    }
                }
                return sums;
            }

            /// @brief Returns whether two equally long vectors hold the same bytes.
            template<typename T>
            bool same_bits(const std::vector<T>& a, const std::vector<T>& b)
            {
                return a.size() == b.size() &&
                       std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
            }

            /// @brief Returns value as T, with a zero imaginary part for a complex T.
            template<typename T>
            T make_value(double value)
            {
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                              std::is_same_v<T, std::complex<double>>)
                    return T(static_cast<typename T::value_type>(value), 0);
                else
                    return static_cast<T>(value);
            }

            /// @brief The weights a case with n inputs uses: the first n of K_WEIGHTS, or all 1 for
            ///        an integer type.
            template<typename T>
            std::vector<double> weights_for(std::size_t n)
            {
                if constexpr (std::is_integral_v<T>)
                    return std::vector<double>(n, 1.0);
                else
                    return std::vector<double>(K_WEIGHTS.begin(),
                                               K_WEIGHTS.begin() + static_cast<std::ptrdiff_t>(n));
            }
        } // namespace

        /**
         * @brief Runs ordered_sum for one element type.
         * @tparam T One of the seven element types.
         */
        template<typename T>
        class Medium_OrderedSumTest : public SushiBLASTest
        {
            protected:
                /// @brief Uploads each vector as a K_ROWS x K_COLUMNS tensor.
                std::vector<Tensor> upload(const std::vector<std::vector<T>>& host)
                {
                    std::vector<Tensor> tensors;
                    for (const std::vector<T>& values : host)
                        tensors.push_back(this->template create_initialized_tensor<T>(
                            {K_ROWS, K_COLUMNS}, values));
                    return tensors;
                }

                /// @brief Records the sum under mode, runs it and returns the output.
                std::vector<T> run_sum(ExecutionMode mode, const std::vector<Tensor>& inputs,
                                       const std::vector<double>& weights)
                {
                    this->set_mode(mode);
                    Tensor out =
                        this->engine->create_tensor({K_ROWS, K_COLUMNS}, sushiblas_dtype<T>());
                    SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(inputs, weights, out);
                    this->recording->run().value();
                    return read_tensor<T>(out);
                }

                /// @brief Checks that both paths equal the serial sum bit for bit for n inputs.
                void expect_serial_equality(std::size_t n)
                {
                    std::uint64_t state = K_SEED + n;
                    std::vector<std::vector<T>> host;
                    for (std::size_t k = 0; k < n; ++k)
                        host.push_back(draw_values<T>(state));

                    const std::vector<double> weights = weights_for<T>(n);
                    const std::vector<T> expected = serial_sum<T>(host, weights);
                    const std::vector<Tensor> inputs = upload(host);

                    EXPECT_TRUE(same_bits(expected,
                                          run_sum(ExecutionMode::ForceReference, inputs, weights)))
                        << "the reference path differs from the serial sum for n = " << n;
                    EXPECT_TRUE(
                        same_bits(expected, run_sum(ExecutionMode::ForceSYCL, inputs, weights)))
                        << "the SYCL path differs from the serial sum for n = " << n;
                }

                /// @brief Returns one tensor of this fixture's type and shape.
                Tensor make_tensor()
                {
                    return this->engine->create_tensor({K_ROWS, K_COLUMNS}, sushiblas_dtype<T>());
                }
        };

        /// @brief The seven element types under test.
        using OrderedSumTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                                 std::complex<double>, int32_t, int64_t>;
        TYPED_TEST_SUITE(Medium_OrderedSumTest, OrderedSumTypes, DefaultTypeNames);

        /// @brief Tests that one input is copied through its weight.
        TYPED_TEST(Medium_OrderedSumTest, OneInputEqualsTheSerialSum)
        {
            this->expect_serial_equality(1);
        }

        /// @brief Tests that two inputs add in index order.
        TYPED_TEST(Medium_OrderedSumTest, TwoInputsEqualTheSerialSum)
        {
            this->expect_serial_equality(2);
        }

        /// @brief Tests that five inputs add in index order.
        TYPED_TEST(Medium_OrderedSumTest, FiveInputsEqualTheSerialSum)
        {
            this->expect_serial_equality(5);
        }

        /// @brief Tests the hand-worked values 2 * 1 + 3 * 4 = 14, and 1 + 4 = 5 for an integer
        ///        type.
        TYPED_TEST(Medium_OrderedSumTest, HandWorkedSumIsExact)
        {
            using T = TypeParam;
            const bool integer = std::is_integral_v<T>;
            const std::vector<T> first(K_ELEMENTS, make_value<T>(1.0));
            const std::vector<T> second(K_ELEMENTS, make_value<T>(4.0));
            const std::vector<double> weights = integer ? std::vector<double>{1.0, 1.0}
                                                        : std::vector<double>{2.0, 3.0};
            const std::vector<Tensor> inputs = this->upload({first, second});

            const std::vector<T> got = this->run_sum(ExecutionMode::ForceSYCL, inputs, weights);
            const T expected = make_value<T>(integer ? 5.0 : 14.0);
            EXPECT_TRUE(same_bits(std::vector<T>(K_ELEMENTS, expected), got));
        }

        /// @brief Tests that an empty input list is refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesNoInputs)
        {
            Tensor out = this->make_tensor();
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             std::vector<Tensor>{}, std::vector<double>{}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that a weight count that differs from the input count is refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesAWeightCountThatDiffers)
        {
            const std::vector<Tensor> inputs{this->make_tensor(), this->make_tensor()};
            Tensor out = this->make_tensor();
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             inputs, std::vector<double>{1.0}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that inputs of two shapes are refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesMismatchedShapes)
        {
            const std::vector<Tensor> inputs{
                this->make_tensor(),
                this->engine->create_tensor({K_ROWS, K_COLUMNS + 1}, sushiblas_dtype<TypeParam>())};
            Tensor out = this->make_tensor();
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             inputs, std::vector<double>{1.0, 1.0}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that an output of another shape is refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesAnOutputOfAnotherShape)
        {
            const std::vector<Tensor> inputs{this->make_tensor()};
            Tensor out =
                this->engine->create_tensor({K_ROWS, K_COLUMNS + 1}, sushiblas_dtype<TypeParam>());
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             inputs, std::vector<double>{1.0}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that an input of another type than the first is refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesMismatchedTypes)
        {
            using T = TypeParam;
            const Core::DataType other = std::is_same_v<T, float> ? Core::DataType::Float64
                                                                   : Core::DataType::Float32;
            const std::vector<Tensor> inputs{
                this->make_tensor(), this->engine->create_tensor({K_ROWS, K_COLUMNS}, other)};
            Tensor out = this->make_tensor();
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             inputs, std::vector<double>{1.0, 1.0}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that an output of another type than the inputs is refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesAnOutputOfAnotherType)
        {
            using T = TypeParam;
            const Core::DataType other = std::is_same_v<T, float> ? Core::DataType::Float64
                                                                   : Core::DataType::Float32;
            const std::vector<Tensor> inputs{this->make_tensor()};
            Tensor out = this->engine->create_tensor({K_ROWS, K_COLUMNS}, other);
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             inputs, std::vector<double>{1.0}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that an output that is also an input is refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesAnOutputThatOverlapsAnInput)
        {
            Tensor out = this->make_tensor();
            const std::vector<Tensor> inputs{out};
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             inputs, std::vector<double>{1.0}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that more inputs than the limit are refused.
        TYPED_TEST(Medium_OrderedSumTest, RefusesMoreInputsThanTheLimit)
        {
            const std::vector<Tensor> inputs(ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS + 1,
                                             this->make_tensor());
            Tensor out = this->make_tensor();
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder())
                             .ordered_sum(inputs,
                                          std::vector<double>(
                                              ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS + 1, 1.0),
                                          out),
                         std::invalid_argument);
        }

        /// @brief Tests that the integer types refuse a weight other than 1.
        TYPED_TEST(Medium_OrderedSumTest, IntegerSumRefusesAWeightOtherThanOne)
        {
            if (!std::is_integral_v<TypeParam>)
                GTEST_SKIP() << "Only the integer types refuse a weight.";

            const std::vector<Tensor> inputs{this->make_tensor(), this->make_tensor()};
            Tensor out = this->make_tensor();
            EXPECT_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                             inputs, std::vector<double>{1.0, 2.0}, out),
                         std::invalid_argument);
        }

        /// @brief Tests that the floating and complex types accept a weight other than 1.
        TYPED_TEST(Medium_OrderedSumTest, FloatingSumAcceptsAWeightOtherThanOne)
        {
            if (std::is_integral_v<TypeParam>)
                GTEST_SKIP() << "The integer types take only weight 1.";

            const std::vector<Tensor> inputs{this->make_tensor()};
            Tensor out = this->make_tensor();
            EXPECT_NO_THROW(SushiBLAS::ElementwiseOps(this->recorder()).ordered_sum(
                inputs, std::vector<double>{2.0}, out));
            this->recording->run().value();
        }

        /// @brief Tests that integers wrap modulo 2^N instead of overflowing.
        TYPED_TEST(Medium_OrderedSumTest, IntegerSumWrapsAround)
        {
            using T = TypeParam;
            if (!std::is_integral_v<T>)
                GTEST_SKIP() << "Only the integer types wrap.";

            const std::vector<T> maximum(K_ELEMENTS, std::numeric_limits<T>::max());
            const std::vector<T> one(K_ELEMENTS, make_value<T>(1.0));
            const std::vector<Tensor> inputs = this->upload({maximum, one});

            const std::vector<T> got = this->run_sum(ExecutionMode::ForceSYCL, inputs,
                                                     std::vector<double>{1.0, 1.0});
            EXPECT_TRUE(same_bits(std::vector<T>(K_ELEMENTS, std::numeric_limits<T>::min()), got));
        }
    } // namespace Test
} // namespace SushiBLAS
