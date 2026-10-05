/****************************************************************/
/* test_environment.hpp                                         */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
