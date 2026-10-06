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
