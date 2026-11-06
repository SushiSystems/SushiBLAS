/**************************************************************************/
/* small_test_main.cpp                                                    */
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
 * @file small_test_main.cpp
 * @brief Runs every GoogleTest suite of an all-Small_ program with synchronous logging and no runtime.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiLog/log/log_calls.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/test_runner/test_runner_log_category.hpp>

using SushiBLAS::K_TEST_RUNNER_LOG_CATEGORY;

int main(int argc, char **argv)
{
    // Synchronous logging starts no logger thread, so the program stays on one thread.
    SushiLog::set_log_synchronous(true);

    ::testing::InitGoogleTest(&argc, argv);

    int res = RUN_ALL_TESTS();
    SB_LOG_INFO(K_TEST_RUNNER_LOG_CATEGORY, "All tests finished with result: {}", res);
    return res;
}
