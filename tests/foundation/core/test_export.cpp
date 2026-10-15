/**************************************************************************/
/* test_export.cpp                                                        */
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
 * @file test_export.cpp
 * @brief Tests that SUSHIBLAS_API names the symbol visibility of the platform.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/export.hpp>

/// @brief Turns the expansion of a macro into a string literal.
#define SB_TEST_EXPANDED_TEXT(...) SB_TEST_TEXT(__VA_ARGS__)

/// @brief Turns its argument into a string literal.
#define SB_TEST_TEXT(...) #__VA_ARGS__

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_Export, NamesThePlatformVisibilityAttribute)
        {
            const std::string expansion = SB_TEST_EXPANDED_TEXT(SUSHIBLAS_API);

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef SUSHIBLAS_EXPORTS
            EXPECT_NE(expansion.find("dllexport"), std::string::npos) << expansion;
  #else
            EXPECT_NE(expansion.find("dllimport"), std::string::npos) << expansion;
  #endif
#else
            EXPECT_NE(expansion.find("visibility"), std::string::npos) << expansion;
            EXPECT_NE(expansion.find("default"), std::string::npos) << expansion;
#endif
        }
    } // namespace Test
} // namespace SushiBLAS
