/**************************************************************************/
/* test_add.cpp                                                           */
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
 * @file test_add.cpp
 * @brief Tests ElementwiseOps::add against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <SushiBLAS/tensor/runtime_block.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_AddTest : public SushiBLASTest {};

/**
 * @brief Tests that the recorded add writes the sum of the two operands when the task runs on the device.
 */
TEST_F(Medium_AddTest, AddWritesTheSumOfTheTwoOperands) 
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});
    
    SushiBLAS::ElementwiseOps(recorder()).add(a, b, c);
    recording->run().value();
    
    expect_tensor(c, {3.00000f, 2.50000f, 1.00000f, -1.00000f});
}

namespace
{
    /// @brief Holds floats of shared USM that stand in for memory another owner keeps alive.
    class OwnedFloats
    {
        public:
            /// @brief Allocates the USM and writes values into it.
            explicit OwnedFloats(const std::vector<float>& values)
                : address_(sycl::malloc_shared<float>(values.size(), TestEnvironment::queue(0))),
                  count_(values.size()),
                  token_(SushiRuntime::make_liveness())
            {
                std::copy(values.begin(), values.end(), address_);
            }

            OwnedFloats(const OwnedFloats&) = delete;
            OwnedFloats& operator=(const OwnedFloats&) = delete;

            /// @brief Frees the USM.
            ~OwnedFloats()
            {
                sycl::free(address_, TestEnvironment::queue(0));
            }

            /// @brief Returns the first float.
            float* data() const noexcept { return address_; }

            /// @brief Returns a vector tensor over a Storage that borrows the memory.
            SushiBLAS::Tensor tensor() const
            {
                auto storage = SushiBLAS::borrow_runtime_storage(TestEnvironment::runtime(),
                    TestEnvironment::runtime().interop().borrow<std::byte>(
                        reinterpret_cast<std::byte*>(address_), count_ * sizeof(float),
                        SushiRuntime::DeviceIndex{0}, SushiRuntime::Residency::Shared, token_).value());
                return SushiBLAS::Tensor(std::move(storage), {static_cast<int64_t>(count_)},
                                         SushiBLAS::Core::DataType::Float32);
            }

            /// @brief Lets go of the memory, as its owner does when it frees it.
            void release() { token_.reset(); }

        private:
            /// @brief The USM.
            float* address_;

            /// @brief The number of floats.
            std::size_t count_;

            /// @brief Lives as long as the memory is valid.
            SushiRuntime::LivenessToken token_;
    };
}

/**
 * @brief Tests that the plan reads the two borrowed bases and writes the output's base.
 */
TEST_F(Medium_AddTest, TheBorrowedBasesAreReadsAndTheOutputIsAWriteInThePlan)
{
    const OwnedFloats left({1.0f, 2.0f, -3.0f, 0.5f});
    const OwnedFloats right({2.0f, 0.5f, 4.0f, -1.5f});
    const SushiBLAS::Tensor a = left.tensor();
    const SushiBLAS::Tensor b = right.tensor();
    auto c = engine->create_tensor({4});

    SushiBLAS::ElementwiseOps(recorder()).add(a, b, c);

    SushiRuntime::Result<SushiRuntime::PlanView> plan = recording->plan();
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->nodes.size(), std::size_t{1});
    ASSERT_EQ(plan->nodes[0].reads.size(), std::size_t{2});
    EXPECT_EQ(plan->nodes[0].reads[0].base, static_cast<void*>(left.data()));
    EXPECT_EQ(plan->nodes[0].reads[1].base, static_cast<void*>(right.data()));
    ASSERT_EQ(plan->nodes[0].writes.size(), std::size_t{1});
    EXPECT_EQ(plan->nodes[0].writes[0].base, c.storage->allocation_base());
}

/**
 * @brief Tests that the sum of two borrowed operands is written to a new output tensor.
 */
TEST_F(Medium_AddTest, TheSumOfTwoBorrowedOperandsIsWrittenToTheOutput)
{
    const OwnedFloats left({1.0f, 2.0f, -3.0f, 0.5f});
    const OwnedFloats right({2.0f, 0.5f, 4.0f, -1.5f});
    const SushiBLAS::Tensor a = left.tensor();
    const SushiBLAS::Tensor b = right.tensor();
    auto c = engine->create_tensor({4});

    SushiBLAS::ElementwiseOps(recorder()).add(a, b, c);

    EXPECT_TRUE(recording->run_succeeds());
    expect_tensor(c, {3.00000f, 2.50000f, 1.00000f, -1.00000f});
}

/**
 * @brief Tests that a run reading a borrowed operand whose owner let go is refused as dangling.
 */
TEST_F(Medium_AddTest, ARunReadingABorrowedOperandWhoseOwnerLetGoIsRefusedAsDangling)
{
    OwnedFloats left({1.0f, 2.0f, -3.0f, 0.5f});
    const OwnedFloats right({2.0f, 0.5f, 4.0f, -1.5f});
    const SushiBLAS::Tensor a = left.tensor();
    const SushiBLAS::Tensor b = right.tensor();
    auto c = engine->create_tensor({4});

    SushiBLAS::ElementwiseOps(recorder()).add(a, b, c);
    left.release();

    const SushiRuntime::Result<void> ran = recording->run();
    ASSERT_FALSE(ran);
    EXPECT_EQ(ran.error().code, SushiRuntime::ErrorCode::DanglingResource);
}
