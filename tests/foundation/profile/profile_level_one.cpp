/**************************************************************************/
/* profile_level_one.cpp                                                  */
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
 * @file profile_level_one.cpp
 * @brief Defines the zone writers of a translation unit compiled at profiling level 1.
 * @author Mustafa Garip
 * @see tests/foundation/profile/README.md
 */

#include <SushiBLAS/profile/profile_config.hpp>

#undef SB_PROFILE_LEVEL
#define SB_PROFILE_LEVEL 1

#include <SushiBLAS/profile/profile.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Writes a stage zone, which level 1 compiles in. */
        void write_stage_zone_at_level_one()
        {
            SB_ZONE("profile_level_one.stage");
        }

        /** @brief Writes a fine zone, which level 1 compiles out. */
        void write_fine_zone_at_level_one()
        {
            SB_ZONE_FINE("profile_level_one.fine");
        }
    } // namespace Test
} // namespace SushiBLAS
