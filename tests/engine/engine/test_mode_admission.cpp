/**************************************************************************/
/* test_mode_admission.cpp                                                */
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
 * @file test_mode_admission.cpp
 * @brief Tests that every ModeAdmission adapter keeps the ModeAdmission contract.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/host_mode_admission.hpp>
#include <SushiBLAS/engine/mode_admission.hpp>
#include <SushiBLAS/engine/runtime_mode_admission.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the ModeAdmission contract against an adapter; neither needs a device. */
        template<typename Adapter>
        class Small_ModeAdmissionContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Adapter adapter;
        };

        /// @brief The ModeAdmission adapters.
        using ModeAdmissionAdapters = ::testing::Types<HostModeAdmission, RuntimeModeAdmission>;

        SB_TYPED_TEST_SUITE(Small_ModeAdmissionContract, ModeAdmissionAdapters);

        /** @brief Tests that every engine may start a recorder in ForceReference. */
        TYPED_TEST(Small_ModeAdmissionContract, ForceReferenceIsAdmitted)
        {
            const ModeAdmission& admission = this->adapter;

            EXPECT_TRUE(admission.admits(ExecutionMode::ForceReference));
        }
    } // namespace Test
} // namespace SushiBLAS
