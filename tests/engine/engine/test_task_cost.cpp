/**************************************************************************/
/* test_task_cost.cpp                                                     */
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
 * @file test_task_cost.cpp
 * @brief Tests the access and the cost a recorded node states for an operation, a shape and its operands.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/cost/cost.hpp>
#include <SushiBLAS/cost/cost_shape.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_cost.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Names the matrix product the cost table holds a row for.
            const SushiRuntime::OperationID K_GEMM(
                SushiRuntime::OperationID::hash("blas.lvl3.gemm", 14));

            /// @brief Names the elementwise addition the cost table holds a row for.
            const SushiRuntime::OperationID K_ADD(
                SushiRuntime::OperationID::hash("math.ew.add", 11));

            /** @brief Returns the shape of a FLOAT32 product of an m by k and a k by n matrix. */
            [[nodiscard]] GemmShape gemm_shape(std::int64_t m, std::int64_t n, std::int64_t k)
            {
                GemmShape shape;
                shape.m = m;
                shape.n = n;
                shape.k = k;
                return shape;
            }

            /** @brief Returns the cost of a costed task of the given work. */
            [[nodiscard]] Cost costed_work(std::uint64_t flops, std::uint64_t bytes,
                                           Core::DataType dtype)
            {
                Cost cost;
                cost.flops = flops;
                cost.bytes = bytes;
                cost.dtype = dtype;
                return cost;
            }

            /** @brief Checks that actual names the same bytes of the same allocation as expected. */
            void expect_same_region(const SushiRuntime::BufferRegion& actual,
                                    const SushiRuntime::BufferRegion& expected)
            {
                EXPECT_EQ(actual.base, expected.base);
                EXPECT_EQ(actual.offset, expected.offset);
                EXPECT_EQ(actual.length, expected.length);
            }

            /** @brief Holds USM that stands in for memory another owner keeps alive. */
            class OwnedMemory final
            {
                public:
                    /** @brief Allocates n_bytes of shared USM and a live token for it. */
                    explicit OwnedMemory(std::size_t n_bytes)
                        : address_(sycl::malloc_shared<std::byte>(n_bytes, TestEnvironment::queue(0))),
                          token_(SushiRuntime::make_liveness())
                    {

                    }

                    OwnedMemory(const OwnedMemory&) = delete;
                    OwnedMemory& operator=(const OwnedMemory&) = delete;

                    /** @brief Frees the USM. */
                    ~OwnedMemory()
                    {
                        sycl::free(address_, TestEnvironment::queue(0));
                    }

                    /** @brief Returns a storage that borrows count bytes from the start. */
                    [[nodiscard]] SushiRuntime::sushi_ptr<Storage> borrowed_storage(
                        std::size_t count) const
                    {
                        return SushiRuntime::make_sushi<Storage>(
                            Storage::Borrowed{}, TestEnvironment::runtime(),
                            TestEnvironment::runtime().interop().borrow<std::byte>(
                                address_, count, SushiRuntime::DeviceIndex{0},
                                SushiRuntime::Residency::Shared, token_).value());
                    }

                private:
                    /// @brief The USM.
                    std::byte* address_;

                    /// @brief Lives as long as the memory is valid.
                    SushiRuntime::LivenessToken token_;
            };

            /** @brief Holds one dtype and the class the runtime names for it. */
            struct NodeDTypeRow
            {
                Core::DataType dtype;             /**< The dtype of the cost. */
                SushiRuntime::NodeDType expected; /**< The class the node states. */
            };
        } // namespace

        TEST(Small_TaskAccess, AnEmptySetNamesNoRegionAndNoOwner)
        {
            const OperandSet operands;

            const SushiRuntime::Access access = Internal::task_access(operands);

            EXPECT_TRUE(access.reads.empty());
            EXPECT_TRUE(access.writes.empty());
            EXPECT_TRUE(access.lives.empty());
        }

        /** @brief Builds the access of operand sets over tensors of the fixture's engine. */
        class Medium_TaskAccess : public SushiBLASTest
        {

        };

        TEST_F(Medium_TaskAccess, AWrittenOperandIsNotRead)
        {
            const Tensor a = engine->create_tensor({2, 3});
            const Tensor b = engine->create_tensor({3, 4});
            const Tensor c = engine->create_tensor({2, 4});
            OperandSet operands;
            operands.reads(a).reads(b).writes(c);

            const SushiRuntime::Access access = Internal::task_access(operands);

            ASSERT_EQ(access.reads.size(), std::size_t{2});
            ASSERT_EQ(access.writes.size(), std::size_t{1});
            expect_same_region(access.reads[0], a.region());
            expect_same_region(access.reads[1], b.region());
            expect_same_region(access.writes[0], c.region());
        }

        TEST_F(Medium_TaskAccess, AnUpdatedOperandIsReadAndWritten)
        {
            const Tensor c = engine->create_tensor({2, 4});
            OperandSet operands;
            operands.updates(c);

            const SushiRuntime::Access access = Internal::task_access(operands);

            ASSERT_EQ(access.reads.size(), std::size_t{1});
            ASSERT_EQ(access.writes.size(), std::size_t{1});
            expect_same_region(access.reads[0], c.region());
            expect_same_region(access.writes[0], c.region());
        }

        TEST_F(Medium_TaskAccess, AnOwnedStorageNamesNoOwner)
        {
            const Tensor a = engine->create_tensor({4});
            OperandSet operands;
            operands.reads(a);

            const SushiRuntime::Access access = Internal::task_access(operands);

            EXPECT_TRUE(access.lives.empty());
        }

        TEST_F(Medium_TaskAccess, ABorrowedStorageNamesItsLiveOwner)
        {
            const OwnedMemory memory(128);
            const Tensor borrowed(memory.borrowed_storage(128), {4});
            OperandSet operands;
            operands.writes(borrowed);

            const SushiRuntime::Access access = Internal::task_access(operands);

            ASSERT_EQ(access.lives.size(), std::size_t{1});
            EXPECT_FALSE(access.lives[0].expired());
        }

        /** @brief Costs tasks over tensors of the fixture's engine. */
        class Medium_TaskCost : public SushiBLASTest
        {

        };

        TEST(Small_TaskCost, AnEmptySetAddsNoBytesToTheWork)
        {
            const OperandSet operands;

            const Cost cost = Internal::task_cost(K_ADD, ElementwiseShape{4, Core::DataType::Float32},
                                                  operands);

            EXPECT_EQ(cost.flops, std::uint64_t{4});
            EXPECT_EQ(cost.bytes, std::uint64_t{0});
            EXPECT_TRUE(cost.is_costed());
        }

        TEST_F(Medium_TaskCost, AGemmStatesTwoMNKFlopsAndTheBytesOfItsOperands)
        {
            const Tensor a = engine->create_tensor({2, 3});
            const Tensor b = engine->create_tensor({3, 4});
            const Tensor c = engine->create_tensor({2, 4});
            OperandSet operands;
            operands.reads(a).reads(b).writes(c);

            const Cost cost = Internal::task_cost(K_GEMM, gemm_shape(2, 4, 3), operands);

            EXPECT_EQ(cost.flops, std::uint64_t{48});
            EXPECT_EQ(cost.bytes, std::uint64_t{104});
            EXPECT_EQ(cost.dtype, Core::DataType::Float32);
        }

        TEST_F(Medium_TaskCost, AnUpdatedOperandCountsTwiceInTheBytes)
        {
            const Tensor a = engine->create_tensor({2, 3});
            const Tensor b = engine->create_tensor({3, 4});
            const Tensor c = engine->create_tensor({2, 4});
            OperandSet operands;
            operands.reads(a).reads(b).updates(c);

            const Cost cost = Internal::task_cost(K_GEMM, gemm_shape(2, 4, 3), operands);

            EXPECT_EQ(cost.bytes, std::uint64_t{136});
        }

        TEST_F(Medium_TaskCost, AnElementwiseTaskStatesOneFlopPerElement)
        {
            const Tensor a = engine->create_tensor({4});
            const Tensor b = engine->create_tensor({4});
            const Tensor out = engine->create_tensor({4});
            OperandSet operands;
            operands.reads(a).reads(b).writes(out);

            const Cost cost = Internal::task_cost(K_ADD, ElementwiseShape{4, Core::DataType::Float32},
                                                  operands);

            EXPECT_EQ(cost.flops, std::uint64_t{4});
            EXPECT_EQ(cost.bytes, std::uint64_t{48});
        }

        TEST_F(Medium_TaskCost, ATaskWithNoShapeStaysUncostedWhateverItsOperandsSpan)
        {
            const Tensor out = engine->create_tensor({4});
            OperandSet operands;
            operands.writes(out);

            const Cost cost = Internal::task_cost(K_ADD, CostShape{}, operands);

            EXPECT_FALSE(cost.is_costed());
            EXPECT_EQ(cost.bytes, std::uint64_t{0});
        }

        TEST(Small_NodeCostOf, AnUncostedCostStatesNothing)
        {
            const std::optional<SushiRuntime::NodeCost> node_cost =
                Internal::node_cost_of(Cost::uncosted(UncostedReason::NoShape));

            EXPECT_FALSE(node_cost.has_value());
        }

        TEST(Small_NodeCostOf, ACostedCostCarriesItsFlopsAndBytes)
        {
            const std::optional<SushiRuntime::NodeCost> node_cost =
                Internal::node_cost_of(costed_work(48, 104, Core::DataType::Float32));

            ASSERT_TRUE(node_cost.has_value());
            EXPECT_EQ(node_cost->flops, std::uint64_t{48});
            EXPECT_EQ(node_cost->bytes, std::uint64_t{104});
        }

        /** @brief Maps the dtype of a cost to the class of a runtime node. */
        class Small_NodeCostOfDType : public ::testing::TestWithParam<NodeDTypeRow>
        {

        };

        TEST_P(Small_NodeCostOfDType, AFloatingDtypeIsNamedByItsClassAndAnyOtherIsUnstated)
        {
            const NodeDTypeRow row = GetParam();

            const std::optional<SushiRuntime::NodeCost> node_cost =
                Internal::node_cost_of(costed_work(0, 0, row.dtype));

            ASSERT_TRUE(node_cost.has_value());
            EXPECT_EQ(node_cost->dtype, row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_NodeCostOfDType,
            ::testing::Values(
                NodeDTypeRow{Core::DataType::Float32, SushiRuntime::NodeDType::F32},
                NodeDTypeRow{Core::DataType::Half, SushiRuntime::NodeDType::F16},
                NodeDTypeRow{Core::DataType::Float64, SushiRuntime::NodeDType::Unstated},
                NodeDTypeRow{Core::DataType::Int32, SushiRuntime::NodeDType::Unstated}));
    } // namespace Test
} // namespace SushiBLAS
