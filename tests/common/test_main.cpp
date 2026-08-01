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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
