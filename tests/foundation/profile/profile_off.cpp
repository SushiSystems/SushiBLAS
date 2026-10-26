/**************************************************************************/
/* profile_off.cpp                                                        */
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
 * @file profile_off.cpp
 * @brief Defines a function compiled with the profiler off, which uses every SB_ macro.
 * @author Mustafa Garip
 * @see tests/foundation/profile/README.md
 */

#include <SushiBLAS/profile/profile_config.hpp>

#undef SB_ENABLE_PROFILER
#define SB_ENABLE_PROFILER 0

#include <SushiBLAS/profile/profile.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Uses every macro once and returns the compiled level, which is 0 with the profiler off. */
        int use_every_macro_with_the_profiler_off()
        {
            double only_a_macro_reads = 1.0;
            SB_ZONE("profile_off.stage");
            SB_ZONE_FINE("profile_off.fine");
            SB_ZONE_DETAIL("profile_off.detail");
            SB_VALUE("profile_off.value", "", only_a_macro_reads);
            SB_COUNTER("profile_off.count", "", Add, only_a_macro_reads);
            SB_INSTANT("profile_off.instant");
            return Internal::Profile::K_COMPILED_LEVEL;
        }
    } // namespace Test
} // namespace SushiBLAS
