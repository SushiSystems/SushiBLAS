/**************************************************************************/
/* test_dispatch_checks.cpp                                               */
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
 * @file test_dispatch_checks.cpp
 * @brief Tests the record-time checks of the public Dispatch seam, which decide on values and need no device.
 * @author Mustafa Garip
 */

#include <string>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names a dtype row by the dtype's name. */
            std::string dtype_name(const ::testing::TestParamInfo<Core::DataType>& info)
            {
                return Core::to_string(info.param);
            }
        } // namespace

        /// @brief Runs the default-domain check once per dtype outside it.
        class Small_DispatchDomainCheck : public ::testing::TestWithParam<Core::DataType> {};

        /// @brief Tests that the default domain of a consumer functor is REAL: a complex dtype is refused.
        TEST_P(Small_DispatchDomainCheck, RealOnlyDomainRejectsADtypeOutsideIt)
        {
            EXPECT_THROW(
                Dispatch::require_domain<Dispatch::Domain::Real>(GetParam(), "Test family", "test.real_only"),
                std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(Dtypes, Small_DispatchDomainCheck,
                                 ::testing::Values(Core::DataType::Complex32, Core::DataType::Complex64),
                                 dtype_name);

        /// @brief Tests that a parameter list one slot past the capacity throws instead of dropping a value.
        TEST(Small_DispatchParameterCheck, TooManyTaskParamsThrowsInsteadOfDroppingThem)
        {
            static_assert(TaskDescription::K_MAX_PARAMETERS == 8, "the list below is one longer than the capacity");
            TaskDescription description;

            EXPECT_THROW(Dispatch::set_task_params(
                             description, std::initializer_list<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                             "test.overflow"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
