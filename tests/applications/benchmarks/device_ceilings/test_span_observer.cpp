/**************************************************************************/
/* test_span_observer.cpp                                                 */
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
 * @file test_span_observer.cpp
 * @brief Tests that SpanObserver reports the span from the first dispatch's start to the last one's end.
 * @author Mustafa Garip
 * @see tests/applications/benchmarks/device_ceilings/README.md
 */

#include <gtest/gtest.h>
#include <span_observer.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Builds a record that started and ended at the given times. */
            SushiRuntime::NodeEventRecord record(std::uint64_t start, std::uint64_t end)
            {
                SushiRuntime::NodeEventRecord event;
                event.start_ns = start;
                event.end_ns = end;
                return event;
            }
        } // namespace

        TEST(Small_SpanObserver, TheSpanRunsFromTheFirstStartToTheLastEnd)
        {
            Ceilings::SpanObserver observer;
            SushiRuntime::RunObservation observation;
            observation.events = {record(1000, 1500), record(1200, 4000), record(900, 2000)};

            observer.on_run(observation);

            EXPECT_EQ(observer.take_span_ns(), std::uint64_t{3100});
        }

        TEST(Small_SpanObserver, TakingTheSpanForgetsIt)
        {
            Ceilings::SpanObserver observer;
            SushiRuntime::RunObservation observation;
            observation.events = {record(0, 10)};
            observer.on_run(observation);

            EXPECT_EQ(observer.take_span_ns(), std::uint64_t{10});
            EXPECT_EQ(observer.take_span_ns(), std::uint64_t{0});
        }

        TEST(Small_SpanObserver, ARunWithNoDispatchHasNoSpan)
        {
            Ceilings::SpanObserver observer;

            observer.on_run(SushiRuntime::RunObservation{});

            EXPECT_EQ(observer.take_span_ns(), std::uint64_t{0});
        }

        TEST(Small_SpanObserver, TheLastRunReplacesTheOneBefore)
        {
            Ceilings::SpanObserver observer;
            SushiRuntime::RunObservation first;
            first.events = {record(0, 100)};
            SushiRuntime::RunObservation second;
            second.events = {record(500, 520)};

            observer.on_run(first);
            observer.on_run(second);

            EXPECT_EQ(observer.take_span_ns(), std::uint64_t{20});
        }
    } // namespace Test
} // namespace SushiBLAS
