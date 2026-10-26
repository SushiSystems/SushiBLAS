/**************************************************************************/
/* test_ceilings_options.cpp                                              */
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
 * @file test_ceilings_options.cpp
 * @brief Tests the command line of the device_ceilings program: defaults, every option and every refusal.
 * @author Mustafa Garip
 * @see tests/applications/benchmarks/device_ceilings/README.md
 */

#include <vector>
#include <gtest/gtest.h>
#include <ceilings_options.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Parses the arguments after a program name. */
            Ceilings::ParseResult parse(const std::vector<const char*>& arguments)
            {
                std::vector<const char*> argv = {"device_ceilings"};
                argv.insert(argv.end(), arguments.begin(), arguments.end());
                return Ceilings::parse_options(static_cast<int>(argv.size()), argv.data());
            }
        } // namespace

        TEST(Small_CeilingsOptions, NoArgumentGivesTheDefaultProtocol)
        {
            const Ceilings::ParseResult result = parse({});

            ASSERT_TRUE(result.ok);
            EXPECT_EQ(result.options.device, std::size_t{0});
            EXPECT_EQ(result.options.out, ".");
            EXPECT_EQ(result.options.runs, 5U);
            EXPECT_EQ(result.options.warmup, 2U);
            EXPECT_DOUBLE_EQ(result.options.min_span_ms, 10.0);
            EXPECT_DOUBLE_EQ(result.options.spread_limit, 0.05);
            EXPECT_EQ(result.options.stream_mib, std::size_t{256});
        }

        TEST(Small_CeilingsOptions, EveryOptionTakesItsValue)
        {
            const Ceilings::ParseResult result =
                parse({"--device", "1", "--out", "folder", "--runs", "7", "--warmup", "3",
                       "--min-span-ms", "20.5", "--spread-limit", "0.1", "--stream-mib", "64"});

            ASSERT_TRUE(result.ok) << result.message;
            EXPECT_EQ(result.options.device, std::size_t{1});
            EXPECT_EQ(result.options.out, "folder");
            EXPECT_EQ(result.options.runs, 7U);
            EXPECT_EQ(result.options.warmup, 3U);
            EXPECT_DOUBLE_EQ(result.options.min_span_ms, 20.5);
            EXPECT_DOUBLE_EQ(result.options.spread_limit, 0.1);
            EXPECT_EQ(result.options.stream_mib, std::size_t{64});
        }

        TEST(Small_CeilingsOptions, HelpIsNotAnError)
        {
            const Ceilings::ParseResult result = parse({"--help"});

            EXPECT_FALSE(result.ok);
            EXPECT_TRUE(result.help);
        }

        TEST(Small_CeilingsOptions, RefusesAnUnknownOption)
        {
            const Ceilings::ParseResult result = parse({"--frobnicate", "1"});

            EXPECT_FALSE(result.ok);
            EXPECT_FALSE(result.help);
            EXPECT_NE(result.message.find("--frobnicate"), std::string::npos);
        }

        TEST(Small_CeilingsOptions, RefusesAnOptionWithNoValue)
        {
            const Ceilings::ParseResult result = parse({"--runs"});

            EXPECT_FALSE(result.ok);
            EXPECT_NE(result.message.find("needs a value"), std::string::npos);
        }

        TEST(Small_CeilingsOptions, RefusesAValueOutsideTheRange)
        {
            EXPECT_FALSE(parse({"--runs", "0"}).ok);
            EXPECT_FALSE(parse({"--device", "-1"}).ok);
            EXPECT_FALSE(parse({"--device", "one"}).ok);
            EXPECT_FALSE(parse({"--min-span-ms", "0"}).ok);
            EXPECT_FALSE(parse({"--spread-limit", "-0.1"}).ok);
            EXPECT_FALSE(parse({"--stream-mib", "0"}).ok);
            EXPECT_FALSE(parse({"--out", ""}).ok);
        }

        TEST(Small_CeilingsOptions, AWarmupOfZeroIsAccepted)
        {
            const Ceilings::ParseResult result = parse({"--warmup", "0"});

            ASSERT_TRUE(result.ok) << result.message;
            EXPECT_EQ(result.options.warmup, 0U);
        }
    } // namespace Test
} // namespace SushiBLAS
