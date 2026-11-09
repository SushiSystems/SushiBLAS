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
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
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

            /// @brief The weights a case with n inputs uses: K_WEIGHTS repeated to n entries, or all 1
            ///        for an integer type.
            template<typename T>
            std::vector<double> weights_for(std::size_t n)
            {
                std::vector<double> weights(n, 1.0);
                if constexpr (!std::is_integral_v<T>)
                {
                    for (std::size_t k = 0; k < n; ++k)
                        weights[k] = K_WEIGHTS[k % K_WEIGHTS.size()];
                }
                return weights;
            }

            /// @brief The weights of the hand-worked case: 2 and 3, or 1 and 1 for an integer type.
            template<typename T>
            std::vector<double> hand_weights()
            {
                if constexpr (std::is_integral_v<T>)
                    return {1.0, 1.0};
                else
                    return {2.0, 3.0};
            }

            /// @brief The sum of the hand-worked inputs 1 and 4: 2 * 1 + 3 * 4 = 14, or 1 + 4 = 5.
            template<typename T>
            T hand_total()
            {
                if constexpr (std::is_integral_v<T>)
                    return make_value<T>(5.0);
                else
                    return make_value<T>(14.0);
            }

            /**
             * @brief Names an element type and an input count, one row of the serial-sum table.
             * @tparam Element One of the seven element types.
             * @tparam Inputs How many inputs the sum takes.
             */
            template<typename Element, std::size_t Inputs>
            struct SerialCase
            {
                /// @brief The element type.
                using Type = Element;

                /// @brief The number of inputs.
                static constexpr std::size_t K_INPUTS = Inputs;
            };

            /// @brief The input counts of the table: one, two, five and the limit.
            template<typename Element>
            using SerialCounts =
                ::testing::Types<SerialCase<Element, 1>, SerialCase<Element, 2>,
                                 SerialCase<Element, 5>,
                                 SerialCase<Element, ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS>>;

            /// @brief Joins two type lists.
            template<typename First, typename Second>
            struct Joined;

            /// @brief Joins two type lists.
            template<typename... A, typename... B>
            struct Joined<::testing::Types<A...>, ::testing::Types<B...>>
            {
                /// @brief The joined list.
                using type = ::testing::Types<A..., B...>;
            };
        } // namespace

        /**
         * @brief Runs ordered_sum for one element type.
         * @tparam T One of the seven element types.
         */
        template<typename T>
        class OrderedSumFixture : public SushiBLASTest
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
        };

        /// @brief Runs one row of the serial-sum table: an element type and an input count.
        template<typename Case>
        class Medium_OrderedSumSerialTest : public OrderedSumFixture<typename Case::Type> {};

        /// @brief Runs the cases that hold for each of the seven element types.
        template<typename T>
        class Medium_OrderedSumTest : public OrderedSumFixture<T> {};

        /// @brief Runs the cases of the real and complex types, which take any weight.
        template<typename T>
        class Medium_OrderedSumWeightedTest : public OrderedSumFixture<T> {};

        /// @brief Runs the cases of the integer types, which take weight 1 and wrap.
        template<typename T>
        class Medium_OrderedSumIntegerTest : public OrderedSumFixture<T> {};

        /// @brief The seven element types under test.
        using OrderedSumTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                                 std::complex<double>, int32_t, int64_t>;

        /// @brief The real and complex element types.
        using WeightedTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                               std::complex<double>>;

        /// @brief The integer element types.
        using IntegerTypes = ::testing::Types<int32_t, int64_t>;

        /// @brief Every element type at every input count of the table.
        using SerialRows = typename Joined<
            typename Joined<
                typename Joined<SerialCounts<sycl::half>, SerialCounts<float>>::type,
                typename Joined<SerialCounts<double>, SerialCounts<std::complex<float>>>::type>::type,
            typename Joined<
                typename Joined<SerialCounts<std::complex<double>>, SerialCounts<int32_t>>::type,
                SerialCounts<int64_t>>::type>::type;

        TYPED_TEST_SUITE(Medium_OrderedSumSerialTest, SerialRows, DefaultTypeNames);
        TYPED_TEST_SUITE(Medium_OrderedSumTest, OrderedSumTypes, DefaultTypeNames);
        TYPED_TEST_SUITE(Medium_OrderedSumWeightedTest, WeightedTypes, DefaultTypeNames);
        TYPED_TEST_SUITE(Medium_OrderedSumIntegerTest, IntegerTypes, DefaultTypeNames);

        /// @brief Tests that a sum of the row's input count equals the serial sum on both paths.
        TYPED_TEST(Medium_OrderedSumSerialTest, ASumEqualsTheSerialSumBitForBit)
        {
            this->expect_serial_equality(TypeParam::K_INPUTS);
        }

        /// @brief Tests the hand-worked values 2 * 1 + 3 * 4 = 14, and 1 + 4 = 5 for an integer type.
        TYPED_TEST(Medium_OrderedSumTest, HandWorkedSumIsExact)
        {
            using T = TypeParam;
            const std::vector<T> first(K_ELEMENTS, make_value<T>(1.0));
            const std::vector<T> second(K_ELEMENTS, make_value<T>(4.0));
            const std::vector<Tensor> inputs = this->upload({first, second});

            const std::vector<T> got =
                this->run_sum(ExecutionMode::ForceSYCL, inputs, hand_weights<T>());
            EXPECT_TRUE(same_bits(std::vector<T>(K_ELEMENTS, hand_total<T>()), got));
        }

        /// @brief Tests that a weight of 2 on one input of 3 gives 6.
        TYPED_TEST(Medium_OrderedSumWeightedTest, AWeightOfTwoDoublesItsInput)
        {
            using T = TypeParam;
            const std::vector<T> input(K_ELEMENTS, make_value<T>(3.0));
            const std::vector<Tensor> inputs = this->upload({input});

            const std::vector<T> got =
                this->run_sum(ExecutionMode::ForceSYCL, inputs, std::vector<double>{2.0});
            EXPECT_TRUE(same_bits(std::vector<T>(K_ELEMENTS, make_value<T>(6.0)), got));
        }

        /// @brief Tests that integers wrap modulo 2^N instead of overflowing.
        TYPED_TEST(Medium_OrderedSumIntegerTest, ASumWrapsAround)
        {
            using T = TypeParam;
            const std::vector<T> maximum(K_ELEMENTS, std::numeric_limits<T>::max());
            const std::vector<T> one(K_ELEMENTS, make_value<T>(1.0));
            const std::vector<Tensor> inputs = this->upload({maximum, one});

            const std::vector<T> got = this->run_sum(ExecutionMode::ForceSYCL, inputs,
                                                     std::vector<double>{1.0, 1.0});
            EXPECT_TRUE(same_bits(std::vector<T>(K_ELEMENTS, std::numeric_limits<T>::min()), got));
        }

        /// @brief Records ordered_sum on a host engine; every refusal is decided on counts, shapes, types and regions.
        class Small_OrderedSumRefusal : public HostEngineTest
        {
            protected:
                /// @brief Returns a tensor of the shared shape in the given type.
                Tensor make_tensor(Core::DataType dtype = Core::DataType::Float32)
                {
                    return engine->create_tensor({K_ROWS, K_COLUMNS}, dtype);
                }

                /// @brief Expects the sum to be refused with nothing recorded.
                void expect_refused(const std::vector<Tensor>& inputs, const std::vector<double>& weights,
                                    Tensor& out)
                {
                    EXPECT_THROW(ElementwiseOps(recorder()).ordered_sum(inputs, weights, out),
                                 std::invalid_argument);
                    EXPECT_EQ(recording->recorded_count(), 0u);
                }
        };

        /// @brief Tests that an empty input list is refused.
        TEST_F(Small_OrderedSumRefusal, NoInputsAreRefused)
        {
            Tensor out = make_tensor();

            expect_refused({}, {}, out);
        }

        /// @brief Tests that a weight count that differs from the input count is refused.
        TEST_F(Small_OrderedSumRefusal, AWeightCountThatDiffersIsRefused)
        {
            Tensor out = make_tensor();

            expect_refused({make_tensor(), make_tensor()}, {1.0}, out);
        }

        /// @brief Tests that inputs of two shapes are refused.
        TEST_F(Small_OrderedSumRefusal, MismatchedInputShapesAreRefused)
        {
            Tensor out = make_tensor();
            const Tensor wider = engine->create_tensor({K_ROWS, K_COLUMNS + 1}, Core::DataType::Float32);

            expect_refused({make_tensor(), wider}, {1.0, 1.0}, out);
        }

        /// @brief Tests that an output of another shape is refused.
        TEST_F(Small_OrderedSumRefusal, AnOutputOfAnotherShapeIsRefused)
        {
            Tensor out = engine->create_tensor({K_ROWS, K_COLUMNS + 1}, Core::DataType::Float32);

            expect_refused({make_tensor()}, {1.0}, out);
        }

        /// @brief Tests that an input of another type than the first is refused.
        TEST_F(Small_OrderedSumRefusal, MismatchedInputTypesAreRefused)
        {
            Tensor out = make_tensor();

            expect_refused({make_tensor(), make_tensor(Core::DataType::Int32)}, {1.0, 1.0}, out);
        }

        /// @brief Tests that an output of another type than the inputs is refused.
        TEST_F(Small_OrderedSumRefusal, AnOutputOfAnotherTypeIsRefused)
        {
            Tensor out = make_tensor(Core::DataType::Int32);

            expect_refused({make_tensor()}, {1.0}, out);
        }

        /// @brief Tests that an output that is also an input is refused.
        TEST_F(Small_OrderedSumRefusal, AnOutputThatOverlapsAnInputIsRefused)
        {
            Tensor out = make_tensor();

            expect_refused({out}, {1.0}, out);
        }

        /// @brief Tests that more inputs than the limit are refused.
        TEST_F(Small_OrderedSumRefusal, MoreInputsThanTheLimitAreRefused)
        {
            Tensor out = make_tensor();
            const std::size_t count = ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS + 1;

            expect_refused(std::vector<Tensor>(count, make_tensor()), std::vector<double>(count, 1.0), out);
        }

        /// @brief Tests that the limit itself is accepted.
        TEST_F(Small_OrderedSumRefusal, ExactlyTheLimitOfInputsIsAccepted)
        {
            Tensor out = make_tensor();
            const std::size_t count = ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS;

            EXPECT_NO_THROW(ElementwiseOps(recorder()).ordered_sum(
                std::vector<Tensor>(count, make_tensor()), std::vector<double>(count, 1.0), out));
            EXPECT_EQ(recording->recorded_count(), 1u);
        }

        /// @brief Tests that an integer sum refuses a weight other than 1.
        TEST_F(Small_OrderedSumRefusal, AnIntegerSumRefusesAWeightOtherThanOne)
        {
            Tensor out = make_tensor(Core::DataType::Int32);

            expect_refused({make_tensor(Core::DataType::Int32), make_tensor(Core::DataType::Int32)},
                           {1.0, 2.0}, out);
        }

        /// @brief Tests that a floating sum accepts a weight other than 1.
        TEST_F(Small_OrderedSumRefusal, AFloatingSumAcceptsAWeightOtherThanOne)
        {
            Tensor out = make_tensor();

            EXPECT_NO_THROW(ElementwiseOps(recorder()).ordered_sum(std::vector<Tensor>{make_tensor()},
                                                                 std::vector<double>{2.0}, out));
            EXPECT_EQ(recording->recorded_count(), 1u);
        }
    } // namespace Test
} // namespace SushiBLAS
