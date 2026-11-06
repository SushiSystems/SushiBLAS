/**************************************************************************/
/* status_mapping.hpp                                                     */
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
 * @file status_mapping.hpp
 * @brief Maps every failure of SushiBLAS and SushiRuntime to the host program's Status.
 * @author Mustafa Garip
 */

#pragma once

#include "host_program.hpp"

#include <new>
#include <stdexcept>

#include <SushiBLAS/SushiBLAS.hpp>

namespace HostProgram
{
    /** @brief Holds what the adapter keeps to itself. */
    namespace Detail
    {
        /** @brief Returns the host status for an error a SushiRuntime call returned. */
        [[nodiscard]] inline Status status_of_error(const SushiRuntime::Error& error)
        {
            switch (error.code)
            {
                case SushiRuntime::ErrorCode::InvalidGraph:
                case SushiRuntime::ErrorCode::InvalidAccess:
                case SushiRuntime::ErrorCode::DanglingResource:
                case SushiRuntime::ErrorCode::InvalidState:
                case SushiRuntime::ErrorCode::InvalidArgument:
                    return Status{StatusCode::InvalidRequest, error.message};
                case SushiRuntime::ErrorCode::Unsupported:
                    return Status{StatusCode::Unsupported, error.message};
                case SushiRuntime::ErrorCode::AllocationFailed:
                    return Status{StatusCode::OutOfMemory, error.message};
                case SushiRuntime::ErrorCode::CompileFailed:
                case SushiRuntime::ErrorCode::KernelFailed:
                case SushiRuntime::ErrorCode::DeviceLost:
                case SushiRuntime::ErrorCode::ProtocolMismatch:
                    return Status{StatusCode::BackendFailed, error.message};
            }

            return Status{StatusCode::BackendFailed, error.message};
        }

        /**
         * @brief Runs work and returns its status, or the status of the exception it threw.
         * @tparam Work Callable with no argument that returns a Status.
         */
        template<typename Work>
        [[nodiscard]] Status guarded(const Work& work)
        {
            try
            {
                return work();
            }
            catch (const SushiBLAS::UnsupportedExecution& error)
            {
                return Status{StatusCode::Unsupported, error.what()};
            }
            catch (const std::invalid_argument& error)
            {
                return Status{StatusCode::InvalidRequest, error.what()};
            }
            catch (const std::bad_alloc& error)
            {
                return Status{StatusCode::OutOfMemory, error.what()};
            }
            catch (const std::exception& error)
            {
                return Status{StatusCode::BackendFailed, error.what()};
            }
        }
    } // namespace Detail
} // namespace HostProgram
