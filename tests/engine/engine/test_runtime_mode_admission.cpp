/**************************************************************************/
/* test_runtime_mode_admission.cpp                                        */
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
 * @file test_runtime_mode_admission.cpp
 * @brief Tests that the mode admission of an engine on a runtime admits the modes that run the SYCL kernel.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/runtime_mode_admission.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that an engine on a runtime admits the modes that run the SYCL kernel. */
        TEST(Small_RuntimeModeAdmission, TheModesThatRunTheSYCLKernelAreAdmitted)
        {
            const RuntimeModeAdmission admission{};

            EXPECT_TRUE(admission.admits(ExecutionMode::Auto));
            EXPECT_TRUE(admission.admits(ExecutionMode::ForceSYCL));
        }
    } // namespace Test
} // namespace SushiBLAS
