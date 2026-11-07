/**************************************************************************/
/* test_pair_body.cpp                                                     */
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
 * @file test_pair_body.cpp
 * @brief Tests that a recorded body is called in its current shape or in the old shape that takes the queue.
 * @author Mustafa Garip
 */

#include <vector>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/pair_body.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief A body of the current shape.
            struct CurrentBody
            {
                /** @brief Returns an empty event. */
                sycl::event operator()(const KernelPairDispatcher&,
                                       const std::vector<sycl::event>&) const
                {
                    return sycl::event();
                }
            };

            /// @brief A body of the old shape, which takes the queue.
            struct OldBody
            {
                /** @brief Returns an empty event. */
                sycl::event operator()(const KernelPairDispatcher&, sycl::queue&,
                                       const std::vector<sycl::event>&) const
                {
                    return sycl::event();
                }
            };

            /// @brief A scalar body of the current shape.
            struct CurrentScalarBody
            {
                /** @brief Returns the scalar widened to double. */
                double operator()(float value, const KernelPairDispatcher&,
                                  const std::vector<sycl::event>&) const
                {
                    return static_cast<double>(value) + 1.0;
                }
            };

            /// @brief A scalar body of the old shape.
            struct OldScalarBody
            {
                /** @brief Returns the scalar widened to double. */
                double operator()(float value, const KernelPairDispatcher&, sycl::queue&,
                                  const std::vector<sycl::event>&) const
                {
                    return static_cast<double>(value) + 2.0;
                }
            };
        } // namespace

        /** @brief Tests that a body of the current shape runs with no queue held. */
        TEST(Small_PairBody, ABodyOfTheCurrentShapeRunsWithNoQueue)
        {
            const KernelPairDispatcher dispatcher(ExecutionMode::ForceReference);

            EXPECT_NO_THROW((void)Internal::Detail::call_pair_body(CurrentBody(), dispatcher, {}));
            EXPECT_EQ(Internal::Detail::call_scalar_body(CurrentScalarBody(), 1.0f, dispatcher, {}),
                      2.0);
        }

        /** @brief Tests that a body of the old shape asks the dispatcher for a queue that is not there. */
        TEST(Small_PairBody, ABodyOfTheOldShapeNeedsTheQueueTheDispatcherHolds)
        {
            const KernelPairDispatcher dispatcher(ExecutionMode::ForceReference);

            EXPECT_THROW((void)Internal::Detail::call_pair_body(OldBody(), dispatcher, {}),
                         UnsupportedExecution);
            EXPECT_THROW(
                (void)Internal::Detail::call_scalar_body(OldScalarBody(), 1.0f, dispatcher, {}),
                UnsupportedExecution);
        }

        /** @brief Tests that a body of the old shape runs when the dispatcher holds a queue. */
        TEST(Medium_PairBody, ABodyOfTheOldShapeRunsOnTheHeldQueue)
        {
            const KernelPairDispatcher dispatcher(ExecutionMode::Auto, TestEnvironment::queue());

            EXPECT_NO_THROW((void)Internal::Detail::call_pair_body(OldBody(), dispatcher, {}));
            EXPECT_EQ(Internal::Detail::call_scalar_body(OldScalarBody(), 1.0f, dispatcher, {}),
                      3.0);
        }
    } // namespace Test
} // namespace SushiBLAS
