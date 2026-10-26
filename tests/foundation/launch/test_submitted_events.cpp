/**************************************************************************/
/* test_submitted_events.cpp                                              */
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
 * @file test_submitted_events.cpp
 * @brief Tests that SubmittedEvents keeps every event of a body in order and converts to the last.
 * @author Mustafa Garip
 * @see tests/foundation/launch/README.md
 */

#include <vector>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiBLAS/launch/submitted_events.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_SubmittedEvents, ADefaultValueHoldsNoEvent)
        {
            const Device::SubmittedEvents events;

            EXPECT_TRUE(events.empty());
            EXPECT_TRUE(events.events().empty());
        }

        TEST(Small_SubmittedEvents, AnEventConvertsToAValueHoldingOneEvent)
        {
            const Device::SubmittedEvents events = sycl::event();

            EXPECT_EQ(events.events().size(), std::size_t{1});
        }

        TEST(Small_SubmittedEvents, AppendKeepsSubmissionOrder)
        {
            Device::SubmittedEvents events(sycl::event{});
            events.append(sycl::event{});
            Device::SubmittedEvents later(std::vector<sycl::event>(2));
            events.append(later);

            EXPECT_EQ(events.events().size(), std::size_t{4});
        }

        TEST(Small_SubmittedEvents, AValueConvertsToTheLastEventOrToAnEmptyOne)
        {
            const Device::SubmittedEvents none;
            sycl::event from_none = none;
            const Device::SubmittedEvents some(std::vector<sycl::event>(3));
            sycl::event from_some = some;

            from_none.wait();
            from_some.wait();
            SUCCEED();
        }

        TEST(Small_SubmittedEvents, WaitOnAnEmptyValueReturns)
        {
            const Device::SubmittedEvents none;

            none.wait();
            none.wait_and_throw();
            SUCCEED();
        }
    } // namespace Test
} // namespace SushiBLAS
