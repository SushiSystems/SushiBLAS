/**************************************************************************/
/* test_seeded_schedule.cpp                                               */
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
 * @file test_seeded_schedule.cpp
 * @brief Tests that SeededSchedule runs every step of every task once, each task's steps in order, and the same order for the same seed.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/test_runner/seeded_schedule.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The seed the schedule tests share.
            constexpr std::uint64_t K_SEED = 20261110;

            /// @brief Holds one step that ran: the task it belongs to and its index in the task.
            struct Turn
            {
                std::size_t task;
                std::size_t index;
            };

            /// @brief Adds a task of steps steps that logs each turn into log.
            void add_logging_task(SeededSchedule& schedule, std::size_t task, std::size_t steps,
                                  std::vector<Turn>& log)
            {
                schedule.add_task(steps, [&log, task](std::size_t index)
                                  { log.push_back(Turn{task, index}); });
            }

            /// @brief Returns the indices the given task logged, in the order it logged them.
            std::vector<std::size_t> indices_of(const std::vector<Turn>& log, std::size_t task)
            {
                std::vector<std::size_t> indices;
                for (const Turn& turn : log)
                {
                    if (turn.task == task)
                        indices.push_back(turn.index);
                }
                return indices;
            }
        } // namespace

        TEST(Small_SeededSchedule, EachTasksStepsRunOnceInTheirOwnOrder)
        {
            std::vector<Turn> log;
            SeededSchedule schedule(K_SEED);
            add_logging_task(schedule, 0, 5, log);
            add_logging_task(schedule, 1, 3, log);

            schedule.run();

            EXPECT_EQ(indices_of(log, 0), (std::vector<std::size_t>{0, 1, 2, 3, 4}));
            EXPECT_EQ(indices_of(log, 1), (std::vector<std::size_t>{0, 1, 2}));
        }

        TEST(Small_SeededSchedule, TheOrderNamesTheTaskOfEveryStepThatRan)
        {
            std::vector<Turn> log;
            SeededSchedule schedule(K_SEED);
            add_logging_task(schedule, 0, 2, log);
            add_logging_task(schedule, 1, 2, log);

            schedule.run();

            std::vector<std::size_t> logged;
            for (const Turn& turn : log)
                logged.push_back(turn.task);
            EXPECT_EQ(schedule.order(), logged);
            EXPECT_EQ(std::count(logged.begin(), logged.end(), std::size_t{0}), 2);
            EXPECT_EQ(std::count(logged.begin(), logged.end(), std::size_t{1}), 2);
        }

        TEST(Small_SeededSchedule, TheSameSeedGivesTheSameOrder)
        {
            std::vector<Turn> first_log;
            std::vector<Turn> second_log;
            SeededSchedule first(K_SEED);
            SeededSchedule second(K_SEED);
            add_logging_task(first, 0, 20, first_log);
            add_logging_task(first, 1, 20, first_log);
            add_logging_task(second, 0, 20, second_log);
            add_logging_task(second, 1, 20, second_log);

            first.run();
            second.run();

            EXPECT_EQ(first.order(), second.order());
        }

        TEST(Small_SeededSchedule, AScheduleWithNoTaskRunsNothing)
        {
            SeededSchedule schedule(K_SEED);

            schedule.run();

            EXPECT_TRUE(schedule.order().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
