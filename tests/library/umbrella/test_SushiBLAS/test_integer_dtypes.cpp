/**************************************************************************/
/* test_integer_dtypes.cpp                                                */
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
 * @file test_integer_dtypes.cpp
 * @brief Tests which operations serve INT32 and INT64 tensors and which refuse them, and that every
 *        operation refuses UINT32.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reductions/arg_reduction.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        class Medium_IntegerDTypeTest : public SushiBLASTest
        {
            protected:
                /** @brief Reads a whole INT64 tensor back to the host. */
                std::vector<int64_t> read_i64(const Tensor& t)
                {
                    std::vector<int64_t> host(static_cast<size_t>(t.num_elements));
                    TestEnvironment::queue()
                        .memcpy(host.data(), t.data(), host.size() * sizeof(int64_t)).wait();
                    return host;
                }

                /** @brief Reads a whole INT32 tensor back to the host. */
                std::vector<int32_t> read_i32(const Tensor& t)
                {
                    std::vector<int32_t> host(static_cast<size_t>(t.num_elements));
                    TestEnvironment::queue()
                        .memcpy(host.data(), t.data(), host.size() * sizeof(int32_t)).wait();
                    return host;
                }

                /** @brief Reads an INT32 or INT64 tensor back to the host as INT64. */
                std::vector<int64_t> read_integers(const Tensor& t)
                {
                    if (t.dtype == Core::DataType::Int64)
                        return read_i64(t);

                    const std::vector<int32_t> narrow = read_i32(t);

                    return std::vector<int64_t>(narrow.begin(), narrow.end());
                }

                /** @brief Creates an INT32 or INT64 tensor holding the given values. */
                Tensor filled(const std::vector<float>& values, Core::DataType dtype)
                {
                    Tensor t = engine->create_tensor({static_cast<int64_t>(values.size())}, dtype);
                    fill_tensor(t, values);

                    return t;
                }
        };

        /**
         * @brief Tests that copy moves integer elements.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_IntegerDTypeTest, CopyMovesIntegerElements)
        {
            auto src = engine->create_tensor({3}, Core::DataType::Int64);
            auto dst = engine->create_tensor({3}, Core::DataType::Int64);

            fill_tensor(src, {10.0f, -20.0f, 30.0f});
            SushiBLAS::RandomOps(recorder(), engine->rng()).constant(dst, 0.0);
            SushiBLAS::BLASOps(recorder()).copy(src, dst);
            recording->run().value();

            EXPECT_EQ((std::vector<int64_t>{10, -20, 30}), read_i64(dst));
        }

        /** @brief Holds one binary logic operation, its operands' dtype and the elements it yields. */
        struct BinaryLogicRow
        {
            const char* name;                                         /**< The row name. */
            Core::DataType dtype;                                     /**< The dtype of the operands and the result. */
            std::vector<float> a;                                     /**< The first operand. */
            std::vector<float> b;                                     /**< The second operand. */
            void (*record)(TaskRecorder&, Tensor&, Tensor&, Tensor&); /**< Records the operation on a, b and r. */
            std::vector<int64_t> expected;                            /**< The elements of the result. */
        };

        /** @brief Returns the comparisons and the boolean operations of one promise: they serve integer tensors. */
        static std::vector<BinaryLogicRow> binary_logic_rows()
        {
            return {
                {"Less", Core::DataType::Int32, {1.0f, 5.0f, 3.0f, 9.0f}, {2.0f, 5.0f, 1.0f, 0.0f},
                 [](TaskRecorder& r, Tensor& a, Tensor& b, Tensor& out)
                 { SushiBLAS::LogicOps(r).less(a, b, out); },
                 {1, 0, 0, 0}},
                {"Equal", Core::DataType::Int32, {1.0f, 5.0f, 3.0f, 9.0f}, {2.0f, 5.0f, 1.0f, 0.0f},
                 [](TaskRecorder& r, Tensor& a, Tensor& b, Tensor& out)
                 { SushiBLAS::LogicOps(r).equal(a, b, out); },
                 {0, 1, 0, 0}},
                {"GreaterEqual", Core::DataType::Int32, {1.0f, 5.0f, 3.0f, 9.0f}, {2.0f, 5.0f, 1.0f, 0.0f},
                 [](TaskRecorder& r, Tensor& a, Tensor& b, Tensor& out)
                 { SushiBLAS::LogicOps(r).greater_equal(a, b, out); },
                 {0, 1, 1, 1}},
                {"LogicalAnd", Core::DataType::Int64, {1.0f, 1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 0.0f},
                 [](TaskRecorder& r, Tensor& a, Tensor& b, Tensor& out)
                 { SushiBLAS::LogicOps(r).logical_and(a, b, out); },
                 {1, 0, 0, 0}},
                {"LogicalXor", Core::DataType::Int64, {1.0f, 1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 0.0f},
                 [](TaskRecorder& r, Tensor& a, Tensor& b, Tensor& out)
                 { SushiBLAS::LogicOps(r).logical_xor(a, b, out); },
                 {0, 1, 1, 0}},
            };
        }

        /** @brief Names a binary logic row for the test name. */
        static std::string binary_logic_row_name(const ::testing::TestParamInfo<BinaryLogicRow>& info)
        {
            return info.param.name;
        }

        /** @brief Runs one binary logic operation on integer tensors, one row at a time. */
        class Medium_IntegerBinaryLogic : public Medium_IntegerDTypeTest,
                                          public ::testing::WithParamInterface<BinaryLogicRow>
        {

        };

        TEST_P(Medium_IntegerBinaryLogic, ServesIntegerTensors)
        {
            const BinaryLogicRow& row = GetParam();
            Tensor a = filled(row.a, row.dtype);
            Tensor b = filled(row.b, row.dtype);
            Tensor r = engine->create_tensor({static_cast<int64_t>(row.a.size())}, row.dtype);

            row.record(recorder(), a, b, r);
            recording->run().value();

            const std::vector<int64_t> got = read_integers(r);
            EXPECT_EQ(row.expected, got);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_IntegerBinaryLogic, ::testing::ValuesIn(binary_logic_rows()),
                                 binary_logic_row_name);

        /** @brief Tests that where selects between two INT64 tensors by an INT64 mask. */
        TEST_F(Medium_IntegerDTypeTest, WhereSelectsBetweenIntegerTensors)
        {
            Tensor mask = filled({1.0f, 1.0f, 0.0f, 0.0f}, Core::DataType::Int64);
            Tensor lhs = filled({100.0f, 200.0f, 300.0f, 400.0f}, Core::DataType::Int64);
            Tensor rhs = filled({-1.0f, -2.0f, -3.0f, -4.0f}, Core::DataType::Int64);
            Tensor r = engine->create_tensor({4}, Core::DataType::Int64);

            SushiBLAS::LogicOps(recorder()).where(mask, lhs, rhs, r);
            recording->run().value();

            EXPECT_EQ((std::vector<int64_t>{100, 200, -3, -4}), read_i64(r));
        }

        /** @brief Holds the INT32 truth vectors and the one-element result the all and any tests share. */
        class Medium_IntegerTruthReduction : public Medium_IntegerDTypeTest
        {
            protected:
                /** @brief Returns the one-element INT32 result of a reduction. */
                Tensor result()
                {
                    return engine->create_tensor({1}, Core::DataType::Int32);
                }
        };

        // `all` and `any` reduce a truth vector, so they take integer operands too.
        TEST_F(Medium_IntegerTruthReduction, AllIsZeroWhenOneElementIsZero)
        {
            Tensor mixed = filled({1.0f, 0.0f, 1.0f, 1.0f}, Core::DataType::Int32);
            Tensor r = result();

            SushiBLAS::LogicOps(recorder()).all(mixed, r);
            recording->run().value();

            EXPECT_EQ(0, read_i32(r)[0]);
        }

        TEST_F(Medium_IntegerTruthReduction, AllIsOneWhenEveryElementIsOne)
        {
            Tensor ones = filled({1.0f, 1.0f, 1.0f, 1.0f}, Core::DataType::Int32);
            Tensor r = result();

            SushiBLAS::LogicOps(recorder()).all(ones, r);
            recording->run().value();

            EXPECT_EQ(1, read_i32(r)[0]);
        }

        TEST_F(Medium_IntegerTruthReduction, AnyIsOneWhenOneElementIsOne)
        {
            Tensor mixed = filled({1.0f, 0.0f, 1.0f, 1.0f}, Core::DataType::Int32);
            Tensor r = result();

            SushiBLAS::LogicOps(recorder()).any(mixed, r);
            recording->run().value();

            EXPECT_EQ(1, read_i32(r)[0]);
        }

        /** @brief Holds the INT32 vector the index reductions search and their INT64 result. */
        class Medium_IntegerIndexReduction : public Medium_IntegerDTypeTest
        {
            protected:
                /** @brief Returns the vector 3, -1, 42, 7, 41. */
                Tensor searched()
                {
                    return filled({3.0f, -1.0f, 42.0f, 7.0f, 41.0f}, Core::DataType::Int32);
                }
        };

        /**
         * @brief Tests that argmax over an integer tensor returns an INT64 index.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_IntegerIndexReduction, ArgmaxSearchesAnIntegerTensor)
        {
            Tensor t = searched();
            Tensor r = engine->create_tensor({1}, Core::DataType::Int64);

            SushiBLAS::ReductionOps(recorder()).argmax(t, r);
            recording->run().value();

            EXPECT_EQ(2, read_i64(r)[0]);
        }

        /** @brief Tests that argmin over an integer tensor returns an INT64 index. */
        TEST_F(Medium_IntegerIndexReduction, ArgminSearchesAnIntegerTensor)
        {
            Tensor t = searched();
            Tensor r = engine->create_tensor({1}, Core::DataType::Int64);

            SushiBLAS::ReductionOps(recorder()).argmin(t, r);
            recording->run().value();

            EXPECT_EQ(1, read_i64(r)[0]);
        }

        /**
         * @brief Tests that argmax is exact for an index past 2^24, on a sequence the host generates.
         * @see tests/library/umbrella/README.md
         */
        TEST(Medium_ArgmaxIndex, ArgmaxIsExactPastTheFloatMantissa)
        {
            constexpr std::size_t K_COUNT = 16777218;
            constexpr std::int64_t K_PEAK = 16777217;

            // Every element is 1 and the one at K_PEAK is 2, an index a float cannot name.
            const std::int64_t winner = Dispatch::host_arg_index<float, Reduce::RunningArgMax>(
                K_COUNT, [](std::size_t i) { return i == static_cast<std::size_t>(K_PEAK) ? 2.0f : 1.0f; });

            EXPECT_EQ(K_PEAK, winner)
                << "An index stored as a float rounds to " << static_cast<std::int64_t>(static_cast<float>(K_PEAK))
                << " here, which is a different element.";
        }

        /** @brief The random stream an operation view takes. */
        using RandomStream = std::remove_reference_t<decltype(std::declval<Engine&>().rng())>;

        /** @brief Holds one operation that refuses integer operands. */
        struct RefusalRow
        {
            const char* name; /**< The row name. */
            void (*record)(TaskRecorder&, RandomStream&, Tensor&, Tensor&, Tensor&); /**< Records it on a, b and the scalar. */
        };

        /** @brief Returns the operations that refuse an INT32 operand; `copy` is the exception and is covered above. */
        static std::vector<RefusalRow> refusal_rows()
        {
            return {
                {"Exp", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor&, Tensor&)
                        { SushiBLAS::ElementwiseOps(r).exp(a); }},
                {"Sqrt", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor&, Tensor&)
                         { SushiBLAS::ElementwiseOps(r).sqrt(a); }},
                {"Relu", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor&, Tensor&)
                         { SushiBLAS::NonLinearOps(r).relu(a); }},
                {"Sigmoid", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor&, Tensor&)
                            { SushiBLAS::NonLinearOps(r).sigmoid(a); }},
                {"Uniform", [](TaskRecorder& r, RandomStream& rng, Tensor& a, Tensor&, Tensor&)
                            { SushiBLAS::RandomOps(r, rng).uniform(a, 0.0, 1.0); }},
                {"Normal", [](TaskRecorder& r, RandomStream& rng, Tensor& a, Tensor&, Tensor&)
                           { SushiBLAS::RandomOps(r, rng).normal(a, 0.0, 1.0); }},
                {"Axpy", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor& b, Tensor&)
                         { SushiBLAS::BLASOps(r).axpy(1.0, a, b); }},
                {"Scal", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor&, Tensor&)
                         { SushiBLAS::BLASOps(r).scal(2.0, a); }},
                {"Dot", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor& b, Tensor& scalar)
                        { SushiBLAS::BLASOps(r).dot(a, b, scalar); }},
                {"Nrm2", [](TaskRecorder& r, RandomStream&, Tensor& a, Tensor&, Tensor& scalar)
                         { SushiBLAS::BLASOps(r).nrm2(a, scalar); }},
            };
        }

        /** @brief Names a refusal row for the test name. */
        static std::string refusal_row_name(const ::testing::TestParamInfo<RefusalRow>& info)
        {
            return info.param.name;
        }

        /** @brief Records one operation on INT32 operands of a graph probe, one row at a time. */
        class Medium_IntegerRefusal : public Medium_IntegerDTypeTest,
                                      public ::testing::WithParamInterface<RefusalRow>
        {

        };

        TEST_P(Medium_IntegerRefusal, ThrowsBeforeItRecordsATask)
        {
            GraphProbe probe;
            Tensor a = engine->create_tensor({4}, Core::DataType::Int32);
            Tensor b = engine->create_tensor({4}, Core::DataType::Int32);
            Tensor scalar = engine->create_tensor({1}, Core::DataType::Int32);

            EXPECT_THROW(GetParam().record(probe.recorder(), engine->rng(), a, b, scalar),
                         std::invalid_argument);
            EXPECT_EQ(0u, probe.take_recorded_count())
                << "A rejected op must throw before it records a task, not while executing one.";
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_IntegerRefusal, ::testing::ValuesIn(refusal_rows()),
                                 refusal_row_name);

        /**
         * @brief Tests that the values written to a UINT32 tensor, including one above the INT32
         *        range, are read back.
         * @see tests/library/umbrella/README.md
         */
        TEST_F(Medium_IntegerDTypeTest, AUInt32TensorHoldsItsValues)
        {
            auto t = engine->create_tensor({3}, Core::DataType::UInt32);
            const std::vector<uint32_t> written = {0u, 7u, 4000000000u};

            SushiBLAS::IO().from_vector(t, written);

            EXPECT_EQ(written, SushiBLAS::IO().to_vector<uint32_t>(t));
        }
    } // namespace Test
} // namespace SushiBLAS
