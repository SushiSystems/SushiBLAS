/**************************************************************************/
/* test_broadcast_task.cpp                                                */
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
 * @file test_broadcast_task.cpp
 * @brief Tests that a broadcast task reads the table of each run on the reference and the SYCL path.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/functor_map/broadcast_task.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The extent of the outer index.
        constexpr std::size_t K_OUTER = 3;

        /// @brief The extent of the inner index; not a multiple of any vector width.
        constexpr std::size_t K_INNER = 5;

        /// @brief The elements the matrix operand is read past the base of its allocation.
        constexpr std::size_t K_SKIPPED = 2;

        /** @brief Returns x * v - x, applied element by element. */
        struct ScaleLess
        {
            /** @brief Applies the operation to one element and one vector value. */
            template<typename T>
            T operator()(T x, T v) const
            {
                return x * v - x;
            }
        };

        /** @brief Runs the tasks on the host CPU queue. */
        class Medium_BroadcastTask : public HostDeviceTest
        {

        };

        TEST_F(Medium_BroadcastTask, ATaskOverBoundOperandsReadsTheTableOfEachRunOnBothPaths)
        {
            const SharedArray<float> a(queue(), small_integers<float>(K_OUTER * K_INNER + K_SKIPPED, 1));
            const SharedArray<float> v(queue(), small_integers<float>(K_INNER, 2));
            SharedArray<float> by_reference(queue(), K_OUTER * K_INNER);
            SharedArray<float> by_sycl(queue(), K_OUTER * K_INNER);
            const Dispatch::Detail::BroadcastBinding<1> binding{
                K_OUTER,
                Dispatch::Detail::MapOperand(OperandBinding{2, 0}),
                Dispatch::Detail::MapOperand(OperandBinding{0, K_SKIPPED * sizeof(float)}),
                {Dispatch::Detail::MapOperand(OperandBinding{1, 0})},
                K_INNER,
                {Dispatch::BroadcastIndex::Inner}};
            const auto task = Dispatch::Detail::make_operand_broadcast_task<Dispatch::Domain::All>(
                Core::DataType::Float32, "test", "broadcast", binding, ScaleLess{});
            void* reference_table[3] = {a.get(), v.get(), by_reference.get()};
            void* sycl_table[3] = {a.get(), v.get(), by_sycl.get()};

            task(KernelPairDispatcher(ExecutionMode::ForceReference, queue()), {},
                 SushiRuntime::BoundOperands::of_bytes(reference_table, 3)).wait();
            task(KernelPairDispatcher(ExecutionMode::ForceSYCL, queue()), {},
                 SushiRuntime::BoundOperands::of_bytes(sycl_table, 3)).wait();

            EXPECT_EQ(by_sycl.values(), by_reference.values());
            for (std::size_t o = 0; o < K_OUTER; ++o)
            {
                for (std::size_t n = 0; n < K_INNER; ++n)
                {
                    const float x = a.get()[K_SKIPPED + o * K_INNER + n];
                    EXPECT_EQ(by_reference.get()[o * K_INNER + n], x * v.get()[n] - x);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
