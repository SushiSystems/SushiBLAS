/**************************************************************************/
/* profile_event_reader.hpp                                               */
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
 * @file profile_event_reader.hpp
 * @brief Declares the port that reads profile events and an adapter that holds them in memory.
 * @author Mustafa Garip
 * @see modules/foundation/profile/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace SushiBLAS
{
    namespace Internal::Profile
    {
        /**
         * @brief Reads the profile events a thread has written and discards what it read.
         *
         * Only the reading side is a port. Zones write into SushiProfiler's process-wide collector,
         * so no port stands in front of the writing side.
         */
        class ProfileEventReader
        {
            public:
                virtual ~ProfileEventReader() = default;

                /**
                 * @brief Returns how many events the thread wrote since the last read and discards them.
                 * @param thread_index The index of the thread whose events are read.
                 * @return The number of events read.
                 */
                virtual std::size_t drain_events_of_thread(std::uint32_t thread_index) = 0;
        };

        /**
         * @brief Holds the events a caller hands it and reads them back by thread.
         */
        class InMemoryEventReader final : public ProfileEventReader
        {
            public:
                /**
                 * @brief Adds one event written by a thread.
                 * @param thread_index The index of the thread that wrote the event.
                 */
                void add_event(std::uint32_t thread_index)
                {
                    m_event_threads.push_back(thread_index);
                }

                /**
                 * @brief Returns how many held events the thread wrote and removes them.
                 * @param thread_index The index of the thread whose events are read.
                 * @return The number of events read.
                 */
                std::size_t drain_events_of_thread(std::uint32_t thread_index) override
                {
                    std::size_t read = 0;
                    std::vector<std::uint32_t> kept;
                    for (const std::uint32_t owner : m_event_threads)
                    {
                        if (owner == thread_index)
                            ++read;
                        else
                            kept.push_back(owner);
                    }
                    m_event_threads = std::move(kept);
                    return read;
                }

            private:
                std::vector<std::uint32_t> m_event_threads;
        };
    } // namespace Internal::Profile
} // namespace SushiBLAS
