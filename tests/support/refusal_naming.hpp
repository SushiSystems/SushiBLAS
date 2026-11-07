/**************************************************************************/
/* refusal_naming.hpp                                                     */
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
 * @file refusal_naming.hpp
 * @brief Declares expect_refusal_naming, the expectation that a refusal names the word that
 *        tells the caller what to do instead.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <stdexcept>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Expects record to throw std::invalid_argument whose message contains word.
         * @param word What the message must name, such as a sibling routine or a mode.
         * @tparam Record Callable `() -> void` that records one refused call.
         */
        template<typename Record>
        void expect_refusal_naming(Record&& record, const char* word)
        {
            try
            {
                record();
                ADD_FAILURE() << "the call was accepted; its refusal should name " << word;
            }
            catch (const std::invalid_argument& error)
            {
                const std::string text = error.what();
                EXPECT_NE(text.find(word), std::string::npos) << text;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
