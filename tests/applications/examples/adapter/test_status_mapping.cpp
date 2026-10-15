/**************************************************************************/
/* test_status_mapping.cpp                                                */
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
 * @file test_status_mapping.cpp
 * @brief Tests that the adapter maps every runtime error and every exception to a host status.
 * @author Mustafa Garip
 */

#include <new>
#include <stdexcept>
#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/core/error.hpp>
#include <SushiBLAS/core/errors.hpp>
#include "host_program.hpp"
#include "status_mapping.hpp"

namespace HostProgram
{
    namespace Test
    {
        /** @brief Returns the status code the adapter gives a runtime error code. */
        static StatusCode code_for(SushiRuntime::ErrorCode code)
        {
            return Detail::status_of_error(SushiRuntime::Error(code, "message")).code;
        }

        /** @brief A body that throws one exception. */
        template<typename Exception>
        struct Thrower
        {
            /** @brief Throws the exception. */
            Status operator()() const
            {
                throw Exception("thrown");
            }
        };

        /** @brief A body that runs out of memory. */
        struct AllocationFailure
        {
            /** @brief Throws std::bad_alloc. */
            Status operator()() const
            {
                throw std::bad_alloc();
            }
        };

        /** @brief A body that succeeds. */
        struct Success
        {
            /** @brief Returns the Ok status. */
            Status operator()() const
            {
                return Status{};
            }
        };

        /** @brief Returns the status code the guard gives the exception a body throws. */
        template<typename Exception>
        static StatusCode code_of_throw()
        {
            return Detail::guarded(Thrower<Exception>{}).code;
        }

        TEST(Unit_StatusMapping, EveryRuntimeErrorHasAHostStatus)
        {
            EXPECT_EQ(code_for(SushiRuntime::ErrorCode::InvalidGraph), StatusCode::InvalidRequest);
            EXPECT_EQ(code_for(SushiRuntime::ErrorCode::InvalidState), StatusCode::InvalidRequest);
            EXPECT_EQ(code_for(SushiRuntime::ErrorCode::Unsupported), StatusCode::Unsupported);
            EXPECT_EQ(code_for(SushiRuntime::ErrorCode::AllocationFailed),
                      StatusCode::OutOfMemory);
            EXPECT_EQ(code_for(SushiRuntime::ErrorCode::KernelFailed), StatusCode::BackendFailed);
            EXPECT_EQ(Detail::status_of_error(
                          SushiRuntime::Error(SushiRuntime::ErrorCode::DeviceLost, "lost"))
                          .message,
                      "lost");
        }

        TEST(Unit_StatusMapping, TheGuardNamesTheExceptionItCaught)
        {
            EXPECT_EQ(code_of_throw<SushiBLAS::UnsupportedExecution>(), StatusCode::Unsupported);
            EXPECT_EQ(code_of_throw<std::invalid_argument>(), StatusCode::InvalidRequest);
            EXPECT_EQ(code_of_throw<std::runtime_error>(), StatusCode::BackendFailed);
            EXPECT_EQ(Detail::guarded(AllocationFailure{}).code, StatusCode::OutOfMemory);
            EXPECT_EQ(Detail::guarded(Success{}).code, StatusCode::Ok);
        }
    } // namespace Test
} // namespace HostProgram
