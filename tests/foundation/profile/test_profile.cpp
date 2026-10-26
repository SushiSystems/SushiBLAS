/**************************************************************************/
/* test_profile.cpp                                                       */
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
 * @file test_profile.cpp
 * @brief Tests that every SB_ macro is a statement or an expression a scope can hold.
 * @author Mustafa Garip
 * @see tests/foundation/profile/README.md
 */

#include <SushiBLAS/profile/profile.hpp>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_Profile, EveryMacroIsUsableInOneScope)
        {
            double sample = 0.5;
            {
                SB_ZONE("test_profile.stage");
                SB_ZONE_FINE("test_profile.fine");
                SB_ZONE_DETAIL("test_profile.detail");
                SB_VALUE("test_profile.sample", "", sample);
                SB_COUNTER("test_profile.count", "", Add, sample);
                SB_INSTANT("test_profile.instant");
            }
            SUCCEED();
        }

        TEST(Small_Profile, AValueOnlyAMacroReadsIsNotUnused)
        {
            const double only_read_by_the_macro = 1.0;
            SB_VALUE("test_profile.only_read", "", only_read_by_the_macro);
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
