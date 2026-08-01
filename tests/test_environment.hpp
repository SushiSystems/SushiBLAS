/**************************************************************************/
/* test_environment.hpp                                                   */
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

#pragma once

#include <gtest/gtest.h>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS 
{
    namespace Test
    {
        /**
        * @brief Global environment for SushiBLAS tests.
        * 
        * Handles one-time initialization of the RuntimeContext and provides access to it.
        */
        class TestEnvironment : public ::testing::Environment 
        {
            public:
                void SetUp() override 
                {
                    SB_LOG_INFO("--- Initializing SushiBLAS Test Environment ---");
                    context_ = std::make_unique<SushiRuntime::Execution::RuntimeContext>();
                    
                    SB_LOG_INFO("Test environment ready.");
                }

                void TearDown() override 
                {
                    SB_LOG_INFO("--- Tearing down SushiBLAS Test Environment ---");
                    context_.reset();
                }

                static SushiRuntime::Execution::RuntimeContext& get_context() 
                {
                    return *context_;
                }

            private:
                static inline std::unique_ptr<SushiRuntime::Execution::RuntimeContext> context_;
        };

    } // namespace Test
} // namespace SushiBLAS
