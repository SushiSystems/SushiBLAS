/**************************************************************************/
/* seeded_schedule.hpp                                                    */
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
 * @file seeded_schedule.hpp
 * @brief Declares SeededSchedule, the single-threaded executor that interleaves the steps of several tasks in an order a seed fixes.
 * @author Mustafa Garip
 * @see modules/foundation/test_runner/README.md
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Runs the steps of several tasks one at a time, picking the task of each step from a seeded generator.
         *
         * A race between callers of a lock-guarded type reproduces from the seed: the steps of one
         * task run in their own order, and the seed picks how the tasks interleave. It starts no
         * thread, so a test over it is `Small_`; the same code under real threads stays `Medium_`.
         * @see modules/foundation/test_runner/README.md
         */
        class SeededSchedule
        {
            public:
                /** @brief Builds a schedule with no task whose picks follow @p seed. */
                explicit SeededSchedule(std::uint64_t seed) noexcept : state_(seed) {}

                /**
                 * @brief Adds a task of @p steps steps.
                 * @param steps How many times the task takes a turn.
                 * @param step Runs the turn of the given index, from 0 up to steps - 1.
                 */
                void add_task(std::size_t steps, std::function<void(std::size_t)> step)
                {
                    tasks_.push_back(Task{steps, 0, std::move(step)});
                }

                /**
                 * @brief Runs every step of every task, each task's steps in order.
                 *
                 * An exception a step throws leaves the schedule where it was and reaches the caller.
                 */
                void run()
                {
                    for (;;)
                    {
                        const std::vector<std::size_t> ready = unfinished_tasks();
                        if (ready.empty())
                            return;

                        const std::size_t task = ready[next_pick() % ready.size()];
                        Task& chosen = tasks_[task];
                        chosen.step(chosen.done);
                        ++chosen.done;
                        order_.push_back(task);
                    }
                }

                /** @brief Returns the task each step ran for, in the order the steps ran. */
                [[nodiscard]] const std::vector<std::size_t>& order() const noexcept
                {
                    return order_;
                }

            private:
                /** @brief Holds one task: how many turns it takes, how many it took and what runs a turn. */
                struct Task
                {
                    /// @brief The turns the task takes in all.
                    std::size_t steps;

                    /// @brief The turns it took so far.
                    std::size_t done;

                    /// @brief Runs the turn of a given index.
                    std::function<void(std::size_t)> step;
                };

                /** @brief Returns the tasks that have turns left, in the order they were added. */
                [[nodiscard]] std::vector<std::size_t> unfinished_tasks() const
                {
                    std::vector<std::size_t> ready;
                    for (std::size_t task = 0; task < tasks_.size(); ++task)
                    {
                        if (tasks_[task].done < tasks_[task].steps)
                            ready.push_back(task);
                    }
                    return ready;
                }

                /** @brief Advances the linear congruential generator and returns its high bits. */
                std::uint64_t next_pick() noexcept
                {
                    state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
                    return state_ >> 33;
                }

                /// @brief The generator state; the seed it started from fixes every pick.
                std::uint64_t state_;

                /// @brief The tasks, in the order they were added.
                std::vector<Task> tasks_;

                /// @brief The task of each step that ran.
                std::vector<std::size_t> order_;
        };
    } // namespace Test
} // namespace SushiBLAS
