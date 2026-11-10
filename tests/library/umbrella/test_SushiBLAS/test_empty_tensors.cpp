/**************************************************************************/
/* test_empty_tensors.cpp                                                 */
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
 * @file test_empty_tensors.cpp
 * @brief Tests what elementwise, non-linear, logic and reduction ops do with an empty tensor.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a FLOAT32 tensor of shape {0, 3} with no storage. */
            Tensor empty_tensor()
            {
                return Tensor(SushiRuntime::sushi_ptr<Storage>(), {0, 3});
            }

            /** @brief One call that takes empty operands: its name and the call that records it. */
            struct NoOpRow
            {
                std::string name;
                std::function<void(TaskRecorder&)> record;
            };

            /** @brief One reduction without an identity and the call that records it on an empty operand. */
            struct RefusalRow
            {
                std::string name;
                std::function<void(TaskRecorder&, const Tensor& empty, Tensor& scalar, Tensor& index)> record;
            };

            /** @brief Names a row by its name, which gtest takes as the test's suffix. */
            template<typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }
        } // namespace

        /// @brief Runs the empty-operand cases on a host engine; the record-time decision is made on the shape.
        class Small_EmptyTensorTest : public HostEngineTest
        {
            protected:
                /** @brief Returns a FLOAT32 scalar tensor that starts from @p value. */
                Tensor scalar_of(float value)
                {
                    Tensor scalar = engine->create_tensor({1});
                    fill_tensor(scalar, {value});
                    return scalar;
                }
        };

        /// @brief Records one call per row over empty operands.
        class Small_EmptyTensorNoOp : public Small_EmptyTensorTest, public ::testing::WithParamInterface<NoOpRow> {};

        /// @brief Tests that an operation over empty tensors records no task and does not throw.
        TEST_P(Small_EmptyTensorNoOp, RecordsNothingAndDoesNotThrow)
        {
            EXPECT_NO_THROW(GetParam().record(recorder()));
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        INSTANTIATE_TEST_SUITE_P(
            Operations, Small_EmptyTensorNoOp,
            ::testing::Values(
                NoOpRow{"Exp", [](TaskRecorder& r) { Tensor t = empty_tensor(); ElementwiseOps(r).exp(t); }},
                NoOpRow{"ReLU", [](TaskRecorder& r) { Tensor t = empty_tensor(); NonLinearOps(r).relu(t); }},
                NoOpRow{"LogicalAnd",
                        [](TaskRecorder& r)
                        {
                            Tensor a = empty_tensor();
                            Tensor b = empty_tensor();
                            Tensor out = empty_tensor();
                            LogicOps(r).logical_and(a, b, out);
                        }},
                NoOpRow{"Equal",
                        [](TaskRecorder& r)
                        {
                            Tensor a = empty_tensor();
                            Tensor b = empty_tensor();
                            Tensor out = empty_tensor();
                            LogicOps(r).equal(a, b, out);
                        }},
                NoOpRow{"Convert",
                        [](TaskRecorder& r)
                        {
                            Tensor src = empty_tensor();
                            Tensor dst(SushiRuntime::sushi_ptr<Storage>(), {0, 3}, Core::DataType::Float64);
                            ElementwiseOps(r).convert(src, dst);
                        }}),
            row_name<NoOpRow>);

        /// @brief Tests that empty operands of different shapes still throw.
        TEST_F(Small_EmptyTensorTest, EmptyOperandsMustStillAgreeInShape)
        {
            Tensor a = empty_tensor();
            Tensor b(SushiRuntime::sushi_ptr<Storage>(), {3, 0});
            Tensor c = empty_tensor();

            EXPECT_THROW(ElementwiseOps(recorder()).add(a, b, c), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        /// @brief Tests that all of a storage-less empty tensor writes the identity 1.
        TEST_F(Small_EmptyTensorTest, AllOfAnEmptyTensorWritesOne)
        {
            Tensor t = empty_tensor();
            Tensor out = scalar_of(0.5f);

            EXPECT_NO_THROW(LogicOps(recorder()).all(t, out));
            ASSERT_TRUE(recording->run_succeeds());
            expect_tensor_equal(out, {1.0f});
        }

        /// @brief Tests that any of a storage-less empty tensor writes the identity 0.
        TEST_F(Small_EmptyTensorTest, AnyOfAnEmptyTensorWritesZero)
        {
            Tensor t = empty_tensor();
            Tensor out = scalar_of(0.5f);

            EXPECT_NO_THROW(LogicOps(recorder()).any(t, out));
            ASSERT_TRUE(recording->run_succeeds());
            expect_tensor_equal(out, {0.0f});
        }

        namespace
        {
            /// @brief Records one reduction with an identity over an empty tensor and the identity it must write.
            struct IdentityRow
            {
                std::string name;
                std::function<void(ReductionOps&, const Tensor& empty, Tensor& out)> record;
                float identity;
            };
        } // namespace

        /// @brief Runs the reductions that have an identity over an empty tensor.
        class Small_EmptyTensorIdentity : public Small_EmptyTensorTest, public ::testing::WithParamInterface<IdentityRow> {};

        /// @brief Tests that a reduction with an identity writes it for an empty tensor.
        TEST_P(Small_EmptyTensorIdentity, WritesItsIdentity)
        {
            Tensor t = empty_tensor();
            Tensor out = scalar_of(0.5f);
            ReductionOps reductions(recorder());

            EXPECT_NO_THROW(GetParam().record(reductions, t, out));
            ASSERT_TRUE(recording->run_succeeds());
            expect_tensor_equal(out, {GetParam().identity});
        }

        INSTANTIATE_TEST_SUITE_P(
            Reductions, Small_EmptyTensorIdentity,
            ::testing::Values(
                IdentityRow{"Sum", [](ReductionOps& r, const Tensor& t, Tensor& o) { r.sum(t, o); }, 0.0f},
                IdentityRow{"Asum", [](ReductionOps& r, const Tensor& t, Tensor& o) { r.asum(t, o); }, 0.0f},
                IdentityRow{"Nrm2", [](ReductionOps& r, const Tensor& t, Tensor& o) { r.nrm2(t, o); }, 0.0f},
                IdentityRow{"Prod", [](ReductionOps& r, const Tensor& t, Tensor& o) { r.prod(t, o); }, 1.0f}),
            row_name<IdentityRow>);

        /// @brief Tests that count_nonfinite of an empty tensor writes the count 0 into an INT64 result.
        TEST_F(Small_EmptyTensorTest, CountNonfiniteOfNothingWritesZero)
        {
            Tensor t = empty_tensor();
            Tensor count = engine->create_tensor({1}, Core::DataType::Int64);
            fill_tensor(count, {7.0f});

            EXPECT_NO_THROW(ReductionOps(recorder()).count_nonfinite(t, count));
            ASSERT_TRUE(recording->run_succeeds());
            expect_tensor_equal(count, {0.0f});
        }

        /// @brief Records one reduction without an identity over an empty tensor.
        class Small_EmptyTensorRefusal : public Small_EmptyTensorTest, public ::testing::WithParamInterface<RefusalRow> {};

        /// @brief Tests that a reduction without an identity refuses an empty tensor and records nothing.
        TEST_P(Small_EmptyTensorRefusal, IsRefused)
        {
            Tensor t = empty_tensor();
            Tensor scalar = engine->create_tensor({1});
            Tensor index = engine->create_tensor({1}, Core::DataType::Int64);

            EXPECT_THROW(GetParam().record(recorder(), t, scalar, index), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        INSTANTIATE_TEST_SUITE_P(
            Reductions, Small_EmptyTensorRefusal,
            ::testing::Values(
                RefusalRow{"Mean", [](TaskRecorder& r, const Tensor& t, Tensor& o, Tensor&) { ReductionOps(r).mean(t, o); }},
                RefusalRow{"Max", [](TaskRecorder& r, const Tensor& t, Tensor& o, Tensor&) { ReductionOps(r).max(t, o); }},
                RefusalRow{"Min", [](TaskRecorder& r, const Tensor& t, Tensor& o, Tensor&) { ReductionOps(r).min(t, o); }},
                RefusalRow{"Var", [](TaskRecorder& r, const Tensor& t, Tensor& o, Tensor&) { ReductionOps(r).var(t, o); }},
                RefusalRow{"Std", [](TaskRecorder& r, const Tensor& t, Tensor& o, Tensor&) { ReductionOps(r).std(t, o); }},
                RefusalRow{"Logsumexp",
                           [](TaskRecorder& r, const Tensor& t, Tensor& o, Tensor&) { ReductionOps(r).logsumexp(t, o); }},
                RefusalRow{"Argmax",
                           [](TaskRecorder& r, const Tensor& t, Tensor&, Tensor& i) { ReductionOps(r).argmax(t, i); }},
                RefusalRow{"Argmin",
                           [](TaskRecorder& r, const Tensor& t, Tensor&, Tensor& i) { ReductionOps(r).argmin(t, i); }},
                RefusalRow{"MaxAlong",
                           [](TaskRecorder& r, const Tensor& t, Tensor&, Tensor&)
                           {
                               Tensor per_column = Tensor(SushiRuntime::sushi_ptr<Storage>(), {3});
                               ReductionOps(r).max_along(t, 0, per_column);
                           }},
                RefusalRow{"MinAlong",
                           [](TaskRecorder& r, const Tensor& t, Tensor&, Tensor&)
                           {
                               Tensor per_column = Tensor(SushiRuntime::sushi_ptr<Storage>(), {3});
                               ReductionOps(r).min_along(t, 0, per_column);
                           }},
                RefusalRow{"MeanSpatial",
                           [](TaskRecorder& r, const Tensor&, Tensor&, Tensor&)
                           {
                               Tensor image(SushiRuntime::sushi_ptr<Storage>(), {0, 2, 2, 3});
                               Tensor per_channel(SushiRuntime::sushi_ptr<Storage>(), {1, 3});
                               ReductionOps(r).mean_spatial(image, per_channel);
                           }}),
            row_name<RefusalRow>);

        /// @brief Tests sum_along over an axis of length zero: five sums of nothing.
        TEST_F(Small_EmptyTensorTest, SumAlongOfAnEmptyAxisWritesZeros)
        {
            Tensor no_rows(SushiRuntime::sushi_ptr<Storage>(), {0, 5});
            Tensor sums = engine->create_tensor({5});
            fill_tensor(sums, std::vector<float>(5, 0.5f));

            EXPECT_NO_THROW(ReductionOps(recorder()).sum_along(no_rows, 0, sums));
            ASSERT_TRUE(recording->run_succeeds());
            expect_tensor_equal(sums, std::vector<float>(5, 0.0f));
        }

        /// @brief Tests sum_along into a result of length zero: no position to write, so no task.
        TEST_F(Small_EmptyTensorTest, SumAlongIntoAnEmptyResultRecordsNothing)
        {
            Tensor no_columns(SushiRuntime::sushi_ptr<Storage>(), {5, 0});
            Tensor nothing(SushiRuntime::sushi_ptr<Storage>(), {0});

            EXPECT_NO_THROW(ReductionOps(recorder()).sum_along(no_columns, 0, nothing));
            EXPECT_EQ(recording->recorded_count(), 0u);
        }
    } // namespace Test
} // namespace SushiBLAS
