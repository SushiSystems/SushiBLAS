/**************************************************************************/
/* test_environment.hpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
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
