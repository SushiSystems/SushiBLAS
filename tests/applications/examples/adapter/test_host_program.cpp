/**************************************************************************/
/* test_host_program.cpp                                                  */
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
 * @file test_host_program.cpp
 * @brief Tests the host program's status names and a multiply-add through the SushiBLAS backend.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "host_program.hpp"

namespace HostProgram
{
    namespace Test
    {
        /** @brief Names one status code and the text to_string gives it. */
        struct StatusNameRow
        {
            const char* name;  /**< The row name a failure shows. */
            StatusCode code;   /**< The status code. */
            const char* text;  /**< The name of the enumerator, which to_string returns. */
        };

        /** @brief Returns one row per status code. */
        std::vector<StatusNameRow> status_name_rows()
        {
            return {{"Ok", StatusCode::Ok, "Ok"},
                    {"InvalidRequest", StatusCode::InvalidRequest, "InvalidRequest"},
                    {"Unsupported", StatusCode::Unsupported, "Unsupported"},
                    {"OutOfMemory", StatusCode::OutOfMemory, "OutOfMemory"},
                    {"BackendFailed", StatusCode::BackendFailed, "BackendFailed"}};
        }

        /** @brief Runs one row of status_name_rows(). */
        class Small_HostProgramStatusName : public ::testing::TestWithParam<StatusNameRow>
        {

        };

        TEST_P(Small_HostProgramStatusName, IsTheNameOfTheCode)
        {
            const StatusNameRow row = GetParam();

            EXPECT_STREQ(to_string(row.code), row.text);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_HostProgramStatusName, ::testing::ValuesIn(status_name_rows()),
            [](const ::testing::TestParamInfo<StatusNameRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_HostProgram, ADefaultStatusIsOkWithNoMessage)
        {
            EXPECT_EQ(Status{}.code, StatusCode::Ok);
            EXPECT_TRUE(Status{}.message.empty());
        }

        /** @brief Runs multiply_add on a 2 by 2 product with an all-ones offset. */
        class Medium_HostProgramMultiplyAdd : public ::testing::Test
        {
            protected:
                /** @brief Holds the shifted product of the run. */
                Matrix result;

                /** @brief Holds the sum of the shifted product of the run. */
                float sum = 0;

                /** @brief Runs the call and returns its status. */
                Status run()
                {
                    BackendHandle backend;
                    const Status created = create_backend(backend);
                    if (created.code != StatusCode::Ok)
                        return created;
                    const Matrix a{2, 2, {1, 2, 3, 4}};
                    const Matrix b{2, 2, {0, 1, 1, 0}};
                    const Matrix offset{2, 2, {1, 1, 1, 1}};
                    return multiply_add(*backend, a, b, offset, result, sum);
                }
        };

        TEST_F(Medium_HostProgramMultiplyAdd, GivesTheShiftedProduct)
        {
            const Status status = run();

            ASSERT_EQ(status.code, StatusCode::Ok) << status.message;
            EXPECT_EQ(result.values, (std::vector<float>{3, 2, 5, 4}));
        }

        TEST_F(Medium_HostProgramMultiplyAdd, GivesTheSumOfTheShiftedProduct)
        {
            const Status status = run();

            ASSERT_EQ(status.code, StatusCode::Ok) << status.message;
            EXPECT_EQ(sum, 14.0f);
        }
    } // namespace Test
} // namespace HostProgram
