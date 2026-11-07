/**************************************************************************/
/* test_io_status.cpp                                                     */
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
 * @file test_io_status.cpp
 * @brief Tests that every IOStatus has its own name and that Ok is the zero value.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/io/io_status.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(static_cast<std::uint8_t>(IOStatus::Ok) == 0, "Ok is the zero value.");

        namespace
        {
            /** @brief One status and the name it prints. */
            struct StatusRow
            {
                IOStatus status;
                const char* name;
            };

            /** @brief Returns every status with its name. */
            std::vector<StatusRow> status_rows()
            {
                return {{IOStatus::Ok, "Ok"},
                        {IOStatus::FileUnreadable, "FileUnreadable"},
                        {IOStatus::FileUnwritable, "FileUnwritable"},
                        {IOStatus::MalformedFile, "MalformedFile"},
                        {IOStatus::Truncated, "Truncated"},
                        {IOStatus::ShapeMismatch, "ShapeMismatch"},
                        {IOStatus::DtypeMismatch, "DtypeMismatch"},
                        {IOStatus::LayoutMismatch, "LayoutMismatch"}};
            }
        } // namespace

        /** @brief Runs one status of status_rows(). */
        class Small_IOStatus : public ::testing::TestWithParam<StatusRow>
        {

        };

        TEST_P(Small_IOStatus, IsNamedAfterItsEnumerator)
        {
            const StatusRow row = GetParam();

            EXPECT_EQ(std::string(to_string(row.status)), row.name);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_IOStatus, ::testing::ValuesIn(status_rows()),
            [](const ::testing::TestParamInfo<StatusRow>& info)
            { return std::string(info.param.name); });
    } // namespace Test
} // namespace SushiBLAS
