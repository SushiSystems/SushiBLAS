/**************************************************************************/
/* test_refusals.cpp                                                      */
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
 * @file test_refusals.cpp
 * @brief Tests what the reductions refuse and what they write for an empty operand, on a host engine with no device.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The element count of the lease rows: past the two-stage threshold of a CPU.
            constexpr std::size_t K_TWO_STAGE_SIZE = 100000;

            /// @brief The value an empty reduction's result holds before the fold writes its identity.
            constexpr float K_UNWRITTEN = 7.0f;

            /** @brief Returns the message of the std::invalid_argument record throws, or "" if none. */
            template<typename Record>
            std::string refusal_of(Record&& record)
            {
                try
                {
                    record();
                }
                catch (const std::invalid_argument& refusal)
                {
                    return std::string(refusal.what());
                }
                return std::string();
            }

            /** @brief Names a dtype row by the dtype's name. */
            std::string dtype_name(const ::testing::TestParamInfo<Core::DataType>& info)
            {
                return Core::to_string(info.param);
            }

            /** @brief Names a row by its description, which gtest takes as the test's suffix. */
            template<typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }

            /** @brief Returns a tensor of shape {0, 3} that has no storage. */
            Tensor empty_operand()
            {
                return Tensor(SushiRuntime::sushi_ptr<Storage>(), {0, 3});
            }

            /** @brief One reduction of a real-only family and the dtype it is handed. */
            struct DtypeRow
            {
                const char* name;
                Core::DataType dtype;
                Core::DataType result_dtype;
                void (*record)(ReductionOps&, const Tensor&, Tensor&);
            };

            /** @brief Returns the real-only reductions over the integer and the complex dtypes. */
            std::vector<DtypeRow> dtype_rows()
            {
                return {
                    {"SumOfInt32", Core::DataType::Int32, Core::DataType::Int32,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.sum(t, out); }},
                    {"Nrm2OfInt32", Core::DataType::Int32, Core::DataType::Int32,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.nrm2(t, out); }},
                    {"MeanOfInt32", Core::DataType::Int32, Core::DataType::Int32,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.mean(t, out); }},
                    {"SumOfInt64", Core::DataType::Int64, Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.sum(t, out); }},
                    {"Nrm2OfInt64", Core::DataType::Int64, Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.nrm2(t, out); }},
                    {"MeanOfInt64", Core::DataType::Int64, Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.mean(t, out); }},
                    {"SumOfComplex32", Core::DataType::Complex32, Core::DataType::Complex32,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.sum(t, out); }},
                    {"MaxOfComplex32", Core::DataType::Complex32, Core::DataType::Complex32,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.max(t, out); }},
                    {"CountNonFiniteOfInt32", Core::DataType::Int32, Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.count_nonfinite(t, out); }},
                    {"CountNonFiniteOfInt64", Core::DataType::Int64, Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.count_nonfinite(t, out); }},
                    {"CountNonFiniteOfComplex32", Core::DataType::Complex32, Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.count_nonfinite(t, out); }},
                    {"CountNonFiniteOfComplex64", Core::DataType::Complex64, Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.count_nonfinite(t, out); }},
                };
            }

            /** @brief One call with a malformed operand or result, over a vector of four and a result of three. */
            struct MalformedCall
            {
                const char* name;
                void (*record)(ReductionOps&, const Tensor&, Tensor&);
            };

            /** @brief Returns the calls a malformed operand makes: a rank-1 axis reduction and a non-scalar result. */
            std::vector<MalformedCall> malformed_calls()
            {
                return {
                    {"SumAlongOfARankOneOperand",
                     [](ReductionOps& ops, const Tensor& vector, Tensor& wide) { ops.sum_along(vector, 0, wide); }},
                    {"MaxAlongOfARankOneOperand",
                     [](ReductionOps& ops, const Tensor& vector, Tensor& wide) { ops.max_along(vector, 0, wide); }},
                    {"SumIntoAThreeElementResult",
                     [](ReductionOps& ops, const Tensor& vector, Tensor& wide) { ops.sum(vector, wide); }},
                    {"ArgMaxIntoAThreeElementResult",
                     [](ReductionOps& ops, const Tensor& vector, Tensor& wide) { ops.argmax(vector, wide); }},
                };
            }

            /** @brief One malformed pair of operands for an axis reduction. */
            struct AxisRow
            {
                const char* name;
                std::vector<int64_t> operand;
                int axis;
                std::vector<int64_t> result;
                Core::DataType result_dtype;
            };

            /** @brief Returns the malformations an axis reduction refuses at record time. */
            std::vector<AxisRow> axis_rows()
            {
                const Core::DataType f32 = Core::DataType::Float32;
                return {
                    {"ARankOneOperand", {4}, 0, {3}, f32},
                    {"ARankThreeOperand", {2, 3, 4}, 1, {3}, f32},
                    {"AnAxisPastTheRank", {3, 4}, 2, {3}, f32},
                    {"ANegativeAxis", {3, 4}, -1, {3}, f32},
                    {"AResultTooLongForAxisOne", {3, 4}, 1, {4}, f32},
                    {"AResultTooShortForAxisZero", {3, 4}, 0, {3}, f32},
                    {"AResultOfTheWrongRank", {3, 4}, 1, {3, 1}, f32},
                    {"AResultOfAnotherDtype", {3, 4}, 1, {3}, Core::DataType::Half},
                };
            }

            /** @brief One reduction whose recording under ForceReference must lease no scratch. */
            struct LeaseRow
            {
                const char* name;
                Core::DataType result_dtype;
                void (*record)(ReductionOps&, const Tensor&, Tensor&);
            };

            /** @brief Returns the folding, centred, index and counting seams. */
            std::vector<LeaseRow> lease_rows()
            {
                return {
                    {"Sum", Core::DataType::Float32,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.sum(t, out); }},
                    {"Var", Core::DataType::Float32,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.var(t, out); }},
                    {"ArgMax", Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.argmax(t, out); }},
                    {"CountNonFinite", Core::DataType::Int64,
                     [](ReductionOps& ops, const Tensor& t, Tensor& out) { ops.count_nonfinite(t, out); }},
                };
            }
        } // namespace

        /** @brief Gives each test a host engine and the reductions over its recorder. */
        class Small_ReductionRefusal : public HostEngineTest
        {
        };

        /** @brief Runs one reduction over a dtype it refuses. */
        class Small_ReductionDtype : public Small_ReductionRefusal,
                                     public ::testing::WithParamInterface<DtypeRow>
        {
        };

        /** @brief Tests that a real-only reduction refuses its dtype by name and records nothing. */
        TEST_P(Small_ReductionDtype, IsRefusedByNameAndRecordsNothing)
        {
            const DtypeRow row = GetParam();
            ReductionOps reductions(recorder());
            Tensor t = engine->create_tensor({4}, row.dtype);
            Tensor result = engine->create_tensor({1}, row.result_dtype);

            const std::string message = refusal_of([&] { row.record(reductions, t, result); });

            EXPECT_NE(std::string::npos, message.find(Core::to_string(row.dtype))) << message;
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReductionDtype, ::testing::ValuesIn(dtype_rows()),
                                 row_name<DtypeRow>);

        /** @brief Runs one malformed call. */
        class Small_ReductionMalformed : public Small_ReductionRefusal,
                                         public ::testing::WithParamInterface<MalformedCall>
        {
        };

        /** @brief Tests that a malformed operand is refused with a message and records nothing. */
        TEST_P(Small_ReductionMalformed, IsRefusedAndRecordsNothing)
        {
            const MalformedCall& call = GetParam();
            ReductionOps reductions(recorder());
            Tensor vector = engine->create_tensor({4});
            Tensor wide = engine->create_tensor({3});

            const std::string message = refusal_of([&] { call.record(reductions, vector, wide); });

            EXPECT_FALSE(message.empty());
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReductionMalformed, ::testing::ValuesIn(malformed_calls()),
                                 row_name<MalformedCall>);

        /** @brief Tests that the sum of an empty operand is the identity 0. */
        TEST_F(Small_ReductionRefusal, AnEmptySumIsZero)
        {
            Tensor result = engine->create_tensor({1});
            fill_tensor(result, std::vector<float>{K_UNWRITTEN});

            ReductionOps(recorder()).sum(empty_operand(), result);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(read_tensor<float>(result)[0], 0.0f);
        }

        /** @brief Tests that the product of an empty operand is the identity 1. */
        TEST_F(Small_ReductionRefusal, AnEmptyProductIsOne)
        {
            Tensor result = engine->create_tensor({1});
            fill_tensor(result, std::vector<float>{K_UNWRITTEN});

            ReductionOps(recorder()).prod(empty_operand(), result);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(read_tensor<float>(result)[0], 1.0f);
        }

        /** @brief Tests that the mean of an empty operand is refused. */
        TEST_F(Small_ReductionRefusal, AnEmptyMeanIsRefused)
        {
            const Tensor empty = empty_operand();
            Tensor scalar = engine->create_tensor({1});

            EXPECT_FALSE(refusal_of([&] { ReductionOps(recorder()).mean(empty, scalar); }).empty());
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Tests that the maximum of an empty operand is refused. */
        TEST_F(Small_ReductionRefusal, AnEmptyMaxIsRefused)
        {
            const Tensor empty = empty_operand();
            Tensor scalar = engine->create_tensor({1});

            EXPECT_FALSE(refusal_of([&] { ReductionOps(recorder()).max(empty, scalar); }).empty());
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Tests that count_nonfinite refuses a result of another dtype than INT64, by name. */
        TEST_F(Small_ReductionRefusal, TheCountResultMustBeInt64)
        {
            Tensor t = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor wrong_dtype = engine->create_tensor({1}, Core::DataType::Float32);

            const std::string message =
                refusal_of([&] { ReductionOps(recorder()).count_nonfinite(t, wrong_dtype); });

            EXPECT_NE(std::string::npos, message.find("INT64")) << message;
        }

        /** @brief Tests that count_nonfinite refuses an INT64 result that is not a scalar, by name. */
        TEST_F(Small_ReductionRefusal, TheCountResultMustBeAScalar)
        {
            Tensor t = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor not_scalar = engine->create_tensor({2}, Core::DataType::Int64);

            const std::string message =
                refusal_of([&] { ReductionOps(recorder()).count_nonfinite(t, not_scalar); });

            EXPECT_NE(std::string::npos, message.find("scalar")) << message;
        }

        /** @brief Runs one reduction over a two-stage-sized input. */
        class Small_ReductionLease : public Small_ReductionRefusal,
                                     public ::testing::WithParamInterface<LeaseRow>
        {
        };

        /** @brief Tests that a reduction recorded under ForceReference leases no scratch. */
        TEST_P(Small_ReductionLease, AReferenceReductionLeasesNoScratch)
        {
            Tensor input = engine->create_tensor({static_cast<int64_t>(K_TWO_STAGE_SIZE)});
            Tensor result = engine->create_tensor({1}, GetParam().result_dtype);
            ReductionOps reductions(recorder());
            const std::size_t idle_before = engine->scratch_pool().idle_block_count();

            GetParam().record(reductions, input, result);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(engine->scratch_pool().idle_block_count(), idle_before);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReductionLease, ::testing::ValuesIn(lease_rows()),
                                 row_name<LeaseRow>);

        /** @brief Runs one malformed pair of operands of an axis reduction. */
        class Small_AxisReductionRefusal : public Small_ReductionRefusal,
                                           public ::testing::WithParamInterface<AxisRow>
        {
        };

        /** @brief Tests that max_along refuses a malformed operand when it is recorded. */
        TEST_P(Small_AxisReductionRefusal, MaxAlongRefusesItAtRecordTime)
        {
            Tensor in = engine->create_tensor(GetParam().operand);
            Tensor out = engine->create_tensor(GetParam().result, GetParam().result_dtype);

            EXPECT_THROW(ReductionOps(recorder()).max_along(in, GetParam().axis, out),
                         std::invalid_argument);
        }

        /** @brief Tests that min_along refuses what max_along refuses. */
        TEST_P(Small_AxisReductionRefusal, MinAlongRefusesItAtRecordTime)
        {
            Tensor in = engine->create_tensor(GetParam().operand);
            Tensor out = engine->create_tensor(GetParam().result, GetParam().result_dtype);

            EXPECT_THROW(ReductionOps(recorder()).min_along(in, GetParam().axis, out),
                         std::invalid_argument);
        }

        /** @brief Tests that sum_along refuses what max_along refuses. */
        TEST_P(Small_AxisReductionRefusal, SumAlongRefusesItAtRecordTime)
        {
            Tensor in = engine->create_tensor(GetParam().operand);
            Tensor out = engine->create_tensor(GetParam().result, GetParam().result_dtype);

            EXPECT_THROW(ReductionOps(recorder()).sum_along(in, GetParam().axis, out),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_AxisReductionRefusal, ::testing::ValuesIn(axis_rows()),
                                 row_name<AxisRow>);

        /** @brief Runs one dtype an axis fold cannot serve. */
        class Small_AxisReductionDtype : public Small_ReductionRefusal,
                                         public ::testing::WithParamInterface<Core::DataType>
        {
        };

        /** @brief Tests that max_along refuses a complex or integer dtype. */
        TEST_P(Small_AxisReductionDtype, MaxAlongRefusesIt)
        {
            Tensor x = engine->create_tensor({3, 4}, GetParam());
            Tensor out = engine->create_tensor({3}, GetParam());

            EXPECT_THROW(ReductionOps(recorder()).max_along(x, 1, out), std::invalid_argument);
        }

        /** @brief Tests that sum_along refuses a complex or integer dtype. */
        TEST_P(Small_AxisReductionDtype, SumAlongRefusesIt)
        {
            Tensor x = engine->create_tensor({3, 4}, GetParam());
            Tensor out = engine->create_tensor({3}, GetParam());

            EXPECT_THROW(ReductionOps(recorder()).sum_along(x, 1, out), std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_AxisReductionDtype,
                                 ::testing::Values(Core::DataType::Complex32,
                                                   Core::DataType::Complex64,
                                                   Core::DataType::Int32, Core::DataType::Int64),
                                 dtype_name);
    } // namespace Test
} // namespace SushiBLAS
