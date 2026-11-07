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
 * @file test_status_mapping.cpp
 * @brief Tests that the adapter maps every runtime error and every exception to a host status.
 * @author Mustafa Garip
 */

#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
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

        /** @brief Names one runtime error code and the status code the adapter gives it. */
        struct ErrorCodeRow
        {
            const char* name;                /**< The row name a failure shows. */
            SushiRuntime::ErrorCode error;   /**< The runtime error code. */
            StatusCode status;               /**< The host status code it maps to. */
        };

        /** @brief Returns one row per runtime error code. */
        std::vector<ErrorCodeRow> error_code_rows()
        {
            using Error = SushiRuntime::ErrorCode;
            return {{"InvalidGraph", Error::InvalidGraph, StatusCode::InvalidRequest},
                    {"InvalidAccess", Error::InvalidAccess, StatusCode::InvalidRequest},
                    {"DanglingResource", Error::DanglingResource, StatusCode::InvalidRequest},
                    {"InvalidState", Error::InvalidState, StatusCode::InvalidRequest},
                    {"InvalidArgument", Error::InvalidArgument, StatusCode::InvalidRequest},
                    {"Unsupported", Error::Unsupported, StatusCode::Unsupported},
                    {"AllocationFailed", Error::AllocationFailed, StatusCode::OutOfMemory},
                    {"CompileFailed", Error::CompileFailed, StatusCode::BackendFailed},
                    {"KernelFailed", Error::KernelFailed, StatusCode::BackendFailed},
                    {"DeviceLost", Error::DeviceLost, StatusCode::BackendFailed},
                    {"ProtocolMismatch", Error::ProtocolMismatch, StatusCode::BackendFailed}};
        }

        /** @brief Runs one row of error_code_rows(). */
        class Small_StatusMappingErrorCode : public ::testing::TestWithParam<ErrorCodeRow>
        {

        };

        TEST_P(Small_StatusMappingErrorCode, MapsToItsHostStatus)
        {
            const ErrorCodeRow row = GetParam();

            EXPECT_EQ(code_for(row.error), row.status);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_StatusMappingErrorCode, ::testing::ValuesIn(error_code_rows()),
            [](const ::testing::TestParamInfo<ErrorCodeRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_StatusMapping, ARuntimeErrorKeepsItsMessage)
        {
            const Status status = Detail::status_of_error(
                SushiRuntime::Error(SushiRuntime::ErrorCode::DeviceLost, "lost"));

            EXPECT_EQ(status.message, "lost");
        }

        /** @brief Names one body and the status code the guard gives its outcome. */
        struct GuardRow
        {
            const char* name;                 /**< The row name a failure shows. */
            std::function<StatusCode()> run;  /**< Runs the guard over the body. */
            StatusCode status;                /**< The code the guard returns. */
        };

        /** @brief Returns one row per exception the guard names, a bad allocation and a success. */
        std::vector<GuardRow> guard_rows()
        {
            return {{"UnsupportedExecution", code_of_throw<SushiBLAS::UnsupportedExecution>,
                     StatusCode::Unsupported},
                    {"InvalidArgument", code_of_throw<std::invalid_argument>,
                     StatusCode::InvalidRequest},
                    {"RuntimeError", code_of_throw<std::runtime_error>, StatusCode::BackendFailed},
                    {"BadAllocation", []() { return Detail::guarded(AllocationFailure{}).code; },
                     StatusCode::OutOfMemory},
                    {"NoException", []() { return Detail::guarded(Success{}).code; },
                     StatusCode::Ok}};
        }

        /** @brief Runs one row of guard_rows(). */
        class Small_StatusMappingGuard : public ::testing::TestWithParam<GuardRow>
        {

        };

        TEST_P(Small_StatusMappingGuard, NamesTheOutcomeItSaw)
        {
            const GuardRow row = GetParam();

            EXPECT_EQ(row.run(), row.status);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_StatusMappingGuard, ::testing::ValuesIn(guard_rows()),
            [](const ::testing::TestParamInfo<GuardRow>& info) { return std::string(info.param.name); });
    } // namespace Test
} // namespace HostProgram
