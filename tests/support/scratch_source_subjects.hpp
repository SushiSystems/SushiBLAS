/**************************************************************************/
/* scratch_source_subjects.hpp                                            */
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
 * @file scratch_source_subjects.hpp
 * @brief Declares the subjects a ScratchBlockSource contract runs on: the host source and the runtime source on the shared runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/host_scratch_source.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a host source and lends it as the port. */
        class HostSourceSubject
        {
            public:
                /** @brief Returns the port under test. */
                [[nodiscard]] ScratchBlockSource& port() noexcept { return source_; }

            private:
                /// @brief The adapter under test.
                HostScratchSource source_;
        };

        /** @brief Holds a runtime source over the runtime the test program made and lends it as the port. */
        class RuntimeSourceSubject
        {
            public:
                /** @brief Returns the port under test. */
                [[nodiscard]] ScratchBlockSource& port() noexcept { return source_; }

            private:
                /// @brief The adapter under test.
                RuntimeScratchSource source_{TestEnvironment::runtime()};
        };
    } // namespace Test
} // namespace SushiBLAS
