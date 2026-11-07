/**************************************************************************/
/* collector_event_reader.hpp                                             */
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
 * @file collector_event_reader.hpp
 * @brief Declares the adapter that reads profile events from SushiProfiler's collector.
 * @author Mustafa Garip
 * @see modules/foundation/profile/README.md
 */

#pragma once

#include <SushiBLAS/profile/profile_config.hpp>

#if SB_ENABLE_PROFILER

#include <cstddef>
#include <cstdint>

#include <SushiBLAS/profile/profile_event_reader.hpp>
#include <SushiProfiler/collector/collector.hpp>

namespace SushiBLAS
{
    namespace Internal::Profile
    {
        /**
         * @brief Reads the events of the process-wide SushiProfiler collector.
         */
        class CollectorEventReader final : public ProfileEventReader
        {
            public:
                /**
                 * @brief Returns the index the collector gives the calling thread.
                 * @return The calling thread's index.
                 */
                static std::uint32_t calling_thread() noexcept
                {
                    return SushiProfiler::Collector::thread_index();
                }

                /**
                 * @brief Drains the collector and returns how many events the thread wrote.
                 * @param thread_index The index of the thread whose events are read.
                 * @return The number of events read.
                 */
                std::size_t drain_events_of_thread(std::uint32_t thread_index) override
                {
                    SushiProfiler::DrainBatch batch;
                    SushiProfiler::Collector::instance().drain(batch);
                    for (const SushiProfiler::ThreadDrain& thread : batch.threads)
                    {
                        if (thread.thread_index == thread_index)
                            return thread.events.size();
                    }
                    return 0;
                }
        };
    } // namespace Internal::Profile
} // namespace SushiBLAS

#endif
