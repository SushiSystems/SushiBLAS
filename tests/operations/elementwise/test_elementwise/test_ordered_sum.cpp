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
 * @brief Tests ElementwiseOps::ordered_sum on the device against the rows of ordered_sum_rows.hpp,
 *        bit for bit, and its refusals.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/ordered_sum_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using OrderedSumRows::K_COLUMNS;
            using OrderedSumRows::K_ELEMENTS;
            using OrderedSumRows::K_ROWS;
            using OrderedSumRows::make_value;
            using OrderedSumRows::same_bits;
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
        };

        /// @brief Runs one row of the table: an element type with its inputs, weights and sum.
        template<typename Row>
        class Medium_OrderedSumRows : public OrderedSumFixture<typename Row::Type> {};

        /// @brief Runs the cases of the real and complex types, which take any weight.
        template<typename T>
        class Medium_OrderedSumWeightedTest : public OrderedSumFixture<T> {};

        /// @brief Runs the cases of the integer types, which take weight 1 and wrap.
        template<typename T>
        class Medium_OrderedSumIntegerTest : public OrderedSumFixture<T> {};

        /// @brief The real and complex element types.
        using WeightedTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                               std::complex<double>>;

        /// @brief The integer element types.
        using IntegerTypes = ::testing::Types<int32_t, int64_t>;

        TYPED_TEST_SUITE(Medium_OrderedSumRows, OrderedSumRows::AllRows, DefaultTypeNames);
        TYPED_TEST_SUITE(Medium_OrderedSumWeightedTest, WeightedTypes, DefaultTypeNames);
        TYPED_TEST_SUITE(Medium_OrderedSumIntegerTest, IntegerTypes, DefaultTypeNames);

        /// @brief Tests that the SYCL path writes the bits of the row's expected sum: the serial
        ///        sum for a drawn row, the hand-worked 14 (5 for an integer type) for the hand row.
        TYPED_TEST(Medium_OrderedSumRows, TheSYCLPathWritesTheExpectedSumBitForBit)
        {
            const std::vector<Tensor> inputs = this->upload(TypeParam::inputs());

            EXPECT_TRUE(same_bits(
                TypeParam::expected(),
                this->run_sum(ExecutionMode::ForceSYCL, inputs, TypeParam::weights())));
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
