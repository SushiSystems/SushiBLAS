/**************************************************************************/
/* test_io_status.cpp                                                     */
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
 * @file test_io_status.cpp
 * @brief Tests that every IOStatus has its own name and that Ok is the zero value.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <set>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/io/io_status.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(static_cast<std::uint8_t>(IOStatus::Ok) == 0, "Ok is the zero value.");

        TEST(Small_IOStatus, EveryStatusHasItsOwnName)
        {
            const IOStatus statuses[] = {
                IOStatus::Ok,            IOStatus::FileUnreadable, IOStatus::FileUnwritable,
                IOStatus::MalformedFile, IOStatus::Truncated,      IOStatus::ShapeMismatch,
                IOStatus::DtypeMismatch, IOStatus::LayoutMismatch,
            };
            std::set<std::string> names;

            for (const IOStatus status : statuses)
                names.insert(to_string(status));

            EXPECT_EQ(names.size(), sizeof(statuses) / sizeof(statuses[0]));
            EXPECT_EQ(std::string(to_string(IOStatus::Truncated)), "Truncated");
            EXPECT_EQ(names.count("Unknown"), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
