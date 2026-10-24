/**************************************************************************/
/* test_operation_options.cpp                                             */
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

/**
 * @file test_operation_options.cpp
 * @brief Tests that OperationOptions names a node and states no determinism class unless told to,
 * and the record-time check every operation family shares.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <string_view>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/operation_options.hpp>
#include <SushiRuntime/core/determinism.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_OperationOptions, ANameAloneStatesNoDeterminismClass)
        {
            const OperationOptions options{"pgs_project"};

            EXPECT_EQ(options.name, "pgs_project");
            EXPECT_EQ(options.determinism, SushiRuntime::Determinism::Unstated);
        }

        TEST(Small_OperationOptions, AClassIsTheRuntimesOwn)
        {
            const OperationOptions options{"pgs_project", SushiRuntime::Determinism::Bitwise};

            EXPECT_EQ(options.determinism, SushiRuntime::Determinism::Bitwise);
        }

        TEST(Small_OperationOptions, TheSharedCheckAcceptsEveryClassOfTheRuntime)
        {
            for (const SushiRuntime::Determinism held :
                 {SushiRuntime::Determinism::Unstated, SushiRuntime::Determinism::Bitwise,
                  SushiRuntime::Determinism::Tolerant, SushiRuntime::Determinism::Cosmetic})
            {
                EXPECT_NO_THROW(
                    Dispatch::require_operation_options(OperationOptions{"node", held}, "op"));
            }
        }

        TEST(Small_OperationOptions, TheSharedCheckRefusesAnEmptyOrNullName)
        {
            EXPECT_THROW(Dispatch::require_operation_options(OperationOptions{""}, "op"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_operation_options(OperationOptions{}, "op"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_operation_options(
                             OperationOptions{std::string_view(nullptr, 0)}, "op"),
                         std::invalid_argument);
        }

        TEST(Small_OperationOptions, TheSharedCheckRefusesAClassOutsideTheEnum)
        {
            const OperationOptions options{
                "node", static_cast<SushiRuntime::Determinism>(
                            static_cast<std::uint8_t>(SushiRuntime::Determinism::Cosmetic) + 1)};

            EXPECT_THROW(Dispatch::require_operation_options(options, "op"),
                         std::invalid_argument);
        }

        TEST(Small_OperationOptions, TheMessageNamesTheOperationAndOptions)
        {
            try
            {
                Dispatch::require_operation_options(OperationOptions{""}, "family.verb");
                ADD_FAILURE() << "an empty name was accepted";
            }
            catch (const std::invalid_argument& error)
            {
                const std::string text = error.what();
                EXPECT_NE(text.find("family.verb"), std::string::npos) << text;
                EXPECT_NE(text.find("'options'"), std::string::npos) << text;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
