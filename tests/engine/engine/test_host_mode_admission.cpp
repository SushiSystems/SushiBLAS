/**************************************************************************/
/* test_host_mode_admission.cpp                                           */
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
 * @file test_host_mode_admission.cpp
 * @brief Tests that the mode admission of a host engine admits no mode that needs a SYCL queue.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/host_mode_admission.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that a host engine admits no mode that needs a SYCL queue. */
        TEST(Small_HostModeAdmission, NoModeThatNeedsAQueueIsAdmitted)
        {
            const HostModeAdmission admission{};

            EXPECT_FALSE(admission.admits(ExecutionMode::Auto));
            EXPECT_FALSE(admission.admits(ExecutionMode::ForceSYCL));
        }
    } // namespace Test
} // namespace SushiBLAS
