/**************************************************************************/
/* test_main.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_main.cpp
 * @brief Runs the selected GoogleTest suites with synchronous logging, under the shared
 *        environment unless only Small_ suites are selected.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <string>

#include <gtest/gtest.h>
#include <SushiLog/log/log_calls.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/test_runner/test_runner_log_category.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

using SushiBLAS::K_TEST_RUNNER_LOG_CATEGORY;

namespace
{
    /**
     * @brief Returns whether every positive pattern of a GoogleTest filter names a Small_ suite.
     * @param filter The value of --gtest_filter: positive patterns, then '-' and negative ones.
     */
    bool selects_only_small(const std::string& filter)
    {
        const std::string positive = filter.substr(0, filter.find('-'));
        if (positive.empty())
            return false;
        std::size_t start = 0;
        while (start <= positive.size())
        {
            const std::size_t end = std::min(positive.find(':', start), positive.size());
            if (positive.compare(start, 6, "Small_") != 0)
                return false;
            start = end + 1;
        }
        return true;
    }
} // namespace

int main(int argc, char **argv)
{
    // Force logs to stream synchronously during testing to avoid overlap with GTest output.
    SushiLog::set_log_synchronous(true);

    ::testing::InitGoogleTest(&argc, argv);

    // A Small_ test opens no device and starts no thread, so it runs without the environment.
    if (!selects_only_small(GTEST_FLAG_GET(filter)))
        ::testing::AddGlobalTestEnvironment(new SushiBLAS::Test::TestEnvironment());

    int res = RUN_ALL_TESTS();
    SB_LOG_INFO(K_TEST_RUNNER_LOG_CATEGORY, "All tests finished with result: {}", res);
    return res;
}
