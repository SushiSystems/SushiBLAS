/****************************************************************/
/* test_main.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_main.cpp
 * @brief Runs every GoogleTest suite under the shared environment with synchronous logging.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/core/logger.hpp>
#include "test_environment.hpp"

int main(int argc, char **argv) 
{
    // Force logs to stream synchronously during testing to avoid overlap with GTest output.
    SR_LOGGER_SET_SYNC_MODE(true);
    SB_LOGGER_SET_SYNC_MODE(true);
    
    ::testing::InitGoogleTest(&argc, argv);
    
    // Global environment registration
    ::testing::AddGlobalTestEnvironment(new SushiBLAS::Test::TestEnvironment());
    
    int res = RUN_ALL_TESTS();
    SB_LOG_INFO("All tests finished with result: {}", res);
    return res;
}
