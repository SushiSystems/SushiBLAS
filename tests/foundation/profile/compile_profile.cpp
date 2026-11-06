/**************************************************************************/
/* compile_profile.cpp                                                    */
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

/**
 * @file compile_profile.cpp
 * @brief Checks at compile time that every SB_ macro is a statement a scope can hold, and that a
 *        value only a macro reads draws no unused-variable warning.
 * @author Mustafa Garip
 */

#include <SushiBLAS/profile/profile.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Uses every SB_ macro in one scope; the build compiles it and nothing calls it. */
        [[maybe_unused]] static void use_every_macro_in_one_scope()
        {
            double sample = 0.5;
            SB_ZONE("compile_profile.stage");
            SB_ZONE_FINE("compile_profile.fine");
            SB_ZONE_DETAIL("compile_profile.detail");
            SB_VALUE("compile_profile.sample", "", sample);
            SB_COUNTER("compile_profile.count", "", Add, sample);
            SB_INSTANT("compile_profile.instant");
        }

        /** @brief Holds a value only a macro reads; the build fails if that draws a warning. */
        [[maybe_unused]] static void read_a_value_only_through_a_macro()
        {
            const double only_read_by_the_macro = 1.0;
            SB_VALUE("compile_profile.only_read", "", only_read_by_the_macro);
        }
    } // namespace Test
} // namespace SushiBLAS
