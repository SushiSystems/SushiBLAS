/**************************************************************************/
/* test_host_handle_space.cpp                                             */
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
 * @file test_host_handle_space.cpp
 * @brief Tests that the host handle space is one process-wide LocalHandleSpace with the id of the host blocks.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/tensor/host_handle_space.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_HostHandleSpace, ItIsOneSpaceWhoseIdIsTheHostSpaceId)
        {
            EXPECT_EQ(&host_handle_space(), &host_handle_space());
            EXPECT_EQ(host_handle_space().space_id(), K_HOST_HANDLE_SPACE_ID);
        }

        TEST(Small_HostHandleSpace, ItMintsDistinctLocalBufferHandles)
        {
            const SushiRuntime::MintOutcome first =
                host_handle_space().mint(SushiRuntime::HandleKind::Buffer);
            const SushiRuntime::MintOutcome second =
                host_handle_space().mint(SushiRuntime::HandleKind::Buffer);

            ASSERT_EQ(first.status, SushiRuntime::HandleStatus::Ok);
            ASSERT_EQ(second.status, SushiRuntime::HandleStatus::Ok);
            EXPECT_NE(first.handle, second.handle);
            EXPECT_TRUE(SushiRuntime::LocalHandleSpace::is_local(first.handle));
        }
    } // namespace Test
} // namespace SushiBLAS
