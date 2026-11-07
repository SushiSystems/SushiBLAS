/**************************************************************************/
/* forced_mode_checks.hpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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

/**
 * @file forced_mode_checks.hpp
 * @brief Declares what the operation tests share: the two forced modes, the loop over them, the
 *        trace of a case, the device check of a dtype and the refusal expectation.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The forced modes every case of an operation test runs in.
        inline constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSYCL,
                                                           ExecutionMode::ForceReference};

        /** @brief Returns the name a failure message gives a mode. */
        inline std::string trace_of(ExecutionMode mode)
        {
            return mode == ExecutionMode::ForceReference ? "ForceReference" : "ForceSYCL";
        }

        /** @brief Returns the name a failure message gives the case label run in mode. */
        inline std::string trace_of(std::string_view label, ExecutionMode mode)
        {
            return std::string(label) + " " + trace_of(mode);
        }

        /** @brief Returns "<operation> over <the dtype of T>" for a failure message. */
        template<typename T>
        std::string trace_of(const char* operation)
        {
            return std::string(operation) + " over " + Core::to_string(sushiblas_dtype<T>());
        }

        /** @brief Runs check once for each forced mode, under a trace that names the mode. */
        template<typename Check>
        void in_each_forced_mode(Check&& check)
        {
            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(mode));
                check(mode);
            }
        }

        /** @brief Returns true when the test device can compute in T. */
        template<typename T>
        bool device_computes()
        {
            const sycl::device device = TestEnvironment::queue().get_device();

            if constexpr (std::is_same_v<T, sycl::half>)
                return device.has(sycl::aspect::fp16);
            else if constexpr (std::is_same_v<T, double>)
                return device.has(sycl::aspect::fp64);
            else
                return true;
        }

        /**
         * @brief Expects record to throw std::invalid_argument naming the operation and the operand.
         * @param op The operation id the message must contain.
         * @param operand The operand the message must name between quotes.
         * @tparam Record Callable `() -> void` that records one malformed call.
         */
        template<typename Record>
        void expect_refusal(Record&& record, const char* op, const char* operand)
        {
            try
            {
                record();
                ADD_FAILURE() << op << " accepted a malformed '" << operand << "'";
            }
            catch (const std::invalid_argument& error)
            {
                const std::string text = error.what();
                EXPECT_NE(text.find(op), std::string::npos) << text;
                EXPECT_NE(text.find("'" + std::string(operand) + "'"), std::string::npos) << text;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
