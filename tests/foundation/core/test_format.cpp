/**************************************************************************/
/* test_format.cpp                                                        */
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
 * @file test_format.cpp
 * @brief Tests Core::format: placeholders, escaped braces, width, fill, precision and bases.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/format.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_Format, ReplacesPlaceholdersInOrder)
        {
            EXPECT_EQ(Core::format("{} + {} = {}", 1, 2, 3), "1 + 2 = 3");
            EXPECT_EQ(Core::format("name={}", std::string("gemm")), "name=gemm");
        }

        TEST(Small_Format, ATextWithoutPlaceholdersIsUnchanged)
        {
            EXPECT_EQ(Core::format("no fields here"), "no fields here");
        }

        TEST(Small_Format, DoubledBracesPrintOneBrace)
        {
            EXPECT_EQ(Core::format("{{}} {}", 7), "{} 7");
        }

        TEST(Small_Format, APlaceholderBeyondTheArgumentsPrintsNothing)
        {
            EXPECT_EQ(Core::format("{} and {}", 5), "5 and ");
        }

        TEST(Small_Format, AnUnclosedBraceIsCopiedAsText)
        {
            EXPECT_EQ(Core::format("value {", 1), "value {");
        }

        TEST(Small_Format, WidthPadsOnTheSideTheAlignmentNames)
        {
            EXPECT_EQ(Core::format("[{:<5}]", 42), "[42   ]");
            EXPECT_EQ(Core::format("[{:>5}]", 42), "[   42]");
            EXPECT_EQ(Core::format("[{:^6}]", 42), "[  42  ]");
        }

        TEST(Small_Format, AFillCharacterPrecedesTheAlignment)
        {
            EXPECT_EQ(Core::format("[{:*>5}]", 7), "[****7]");
            EXPECT_EQ(Core::format("[{:-^7}]", "ab"), "[--ab---]");
        }

        TEST(Small_Format, ZeroFillPadsANumberAfterItsSign)
        {
            EXPECT_EQ(Core::format("{:05}", -42), "-0042");
        }

        TEST(Small_Format, PrecisionAndTypeSelectTheFloatNotation)
        {
            EXPECT_EQ(Core::format("{:.2f}", 3.14159), "3.14");
            EXPECT_EQ(Core::format("{:.1e}", 1500.0), "1.5e+03");
        }

        TEST(Small_Format, IntegerTypesSelectTheBase)
        {
            EXPECT_EQ(Core::format("{:x}", 255), "ff");
            EXPECT_EQ(Core::format("{:X}", 255), "FF");
            EXPECT_EQ(Core::format("{:o}", 8), "10");
            EXPECT_EQ(Core::format("{:d}", 12), "12");
        }

        TEST(Small_Format, ABoolPrintsAsAWord)
        {
            EXPECT_EQ(Core::format("{} {}", true, false), "true false");
        }

        TEST(Small_Format, OneFieldDoesNotLeakItsSpecIntoTheNext)
        {
            EXPECT_EQ(Core::format("{:x} {}", 255, 255), "ff 255");
            EXPECT_EQ(Core::format("{:.1f} {}", 2.26, 2.5), "2.3 2.5");
        }
    } // namespace Test
} // namespace SushiBLAS
