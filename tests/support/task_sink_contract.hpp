/**************************************************************************/
/* task_sink_contract.hpp                                                 */
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
 * @file task_sink_contract.hpp
 * @brief Declares the checks every TaskSink adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <utility>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The message the failing body of the contract throws.
        inline constexpr const char* K_SINK_FAILURE_MESSAGE = "the sink contract body failed";

        /// @brief Holds the names the contract tasks record under, which must outlive the record call.
        inline constexpr const char* K_SINK_TASK_NAMES[] = {"sink.first", "sink.second",
                                                            "sink.third"};

        /** @brief Holds what a body saw when it asked its dispatcher for the queue. */
        struct QueueProbe
        {
            /// @brief Whether has_queue() said a queue is held.
            bool held;

            /// @brief Whether queue() threw UnsupportedExecution.
            bool refused;
        };

        /** @brief Asks a dispatcher whether it holds a queue and whether queue() refuses. */
        [[nodiscard]] inline QueueProbe probe_queue_of(const KernelPairDispatcher& pair)
        {
            QueueProbe probe{pair.has_queue(), false};
            try
            {
                (void)pair.queue();
            }
            catch (const UnsupportedExecution&)
            {
                probe.refused = true;
            }
            return probe;
        }

        /** @brief Holds the checks that define a TaskSink; an adapter's suite calls each one. */
        struct TaskSinkContract
        {
            /** @brief Records a task that writes the subject's target and runs body. */
            template<typename Subject, typename Body>
            static void record_step(Subject& subject, const char* name, Body body)
            {
                TaskDescription description;
                description.name = name;
                OperandSet operands;
                operands.writes(subject.target());
                subject.recorder().record_pair(description, operands, "TaskSink contract",
                                               std::move(body));
            }

            /** @brief Records a task whose reference half appends index to order. */
            template<typename Subject>
            static void record_ordered_step(Subject& subject, std::size_t index,
                                            std::vector<std::size_t>* order)
            {
                record_step(subject, K_SINK_TASK_NAMES[index],
                            [index, order](const KernelPairDispatcher& pair,
                                           const std::vector<sycl::event>& deps) -> sycl::event
                            {
                                return pair.run(
                                    deps, [index, order]() { order->push_back(index); },
                                    [](sycl::queue&, const std::vector<sycl::event>&)
                                    {
                                        return sycl::event();
                                    });
                            });
            }

            /** @brief The bodies run in the order they were recorded. */
            template<typename Subject>
            static void expect_record_order_is_run_order(Subject& subject)
            {
                std::vector<std::size_t> order;
                record_ordered_step(subject, 0, &order);
                record_ordered_step(subject, 1, &order);
                record_ordered_step(subject, 2, &order);

                EXPECT_EQ(subject.run(), std::string());
                EXPECT_EQ(order, (std::vector<std::size_t>{0, 1, 2}));
            }

            /** @brief Under ForceReference the reference half of a body runs and its SYCL half does not. */
            template<typename Subject>
            static void expect_reference_half_runs_under_force_reference(Subject& subject)
            {
                int ran = 0;
                int* flag = &ran;
                record_step(subject, K_SINK_TASK_NAMES[0],
                            [flag](const KernelPairDispatcher& pair,
                                   const std::vector<sycl::event>& deps) -> sycl::event
                            {
                                return pair.run(
                                    deps, [flag]() { *flag = 1; },
                                    [flag](sycl::queue&, const std::vector<sycl::event>&)
                                    {
                                        *flag = 2;
                                        return sycl::event();
                                    });
                            });

                EXPECT_EQ(subject.run(), std::string());
                EXPECT_EQ(ran, 1);
            }

            /** @brief A body that throws stops the run, the later bodies do not run, and the failure is reported. */
            template<typename Subject>
            static void expect_failing_body_stops_the_run(Subject& subject)
            {
                std::vector<std::size_t> order;
                record_ordered_step(subject, 0, &order);
                record_step(subject, K_SINK_TASK_NAMES[1],
                            [](const KernelPairDispatcher&,
                               const std::vector<sycl::event>&) -> sycl::event
                            {
                                throw std::runtime_error(K_SINK_FAILURE_MESSAGE);
                            });
                record_ordered_step(subject, 2, &order);

                const std::string failure = subject.run();

                EXPECT_NE(failure.find(K_SINK_FAILURE_MESSAGE), std::string::npos) << failure;
                EXPECT_NE(failure.find(K_SINK_TASK_NAMES[1]), std::string::npos) << failure;
                EXPECT_EQ(order, (std::vector<std::size_t>{0}));
            }

            /** @brief The dispatcher a body gets holds a queue exactly when the adapter has one, and queue() throws otherwise. */
            template<typename Subject>
            static void expect_queue_follows_the_adapter(Subject& subject)
            {
                QueueProbe seen{false, false};
                QueueProbe* seen_out = &seen;
                record_step(subject, K_SINK_TASK_NAMES[0],
                            [seen_out](const KernelPairDispatcher& pair,
                                       const std::vector<sycl::event>&) -> sycl::event
                            {
                                *seen_out = probe_queue_of(pair);
                                return sycl::event();
                            });

                EXPECT_EQ(subject.run(), std::string());
                EXPECT_EQ(seen.held, Subject::K_HAS_QUEUE);
                EXPECT_EQ(seen.refused, !Subject::K_HAS_QUEUE);
            }
        };
    } // namespace Test
} // namespace SushiBLAS
