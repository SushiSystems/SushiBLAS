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
