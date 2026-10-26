/**************************************************************************/
/* span_observer.hpp                                                      */
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
 * @file span_observer.hpp
 * @brief Declares SpanObserver, which keeps the device span of the last run of a runtime.
 * @author Mustafa Garip
 * @see applications/benchmarks/device_ceilings/README.md
 */

#pragma once

#include <mutex>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/report/run_observer.hpp>

namespace SushiBLAS
{
    namespace Ceilings
    {
        /**
         * @brief Times each run of a runtime on the device clock, from the first dispatch's start to the last one's end.
         *
         * Adding it to a runtime turns device timing on, so the span is the one SushiProfiler
         * would show for the same run.
         * @see applications/benchmarks/device_ceilings/README.md
         */
        class SpanObserver final : public SushiRuntime::RunObserver
        {
            public:
                /** @brief Keeps the span of the finished run; a run with no dispatch keeps 0. */
                void on_run(const SushiRuntime::RunObservation& observation) noexcept override
                {
                    std::uint64_t first = UINT64_MAX;
                    std::uint64_t last = 0;
                    for (const SushiRuntime::NodeEventRecord& record : observation.events)
                    {
                        first = std::min(first, record.start_ns);
                        last = std::max(last, record.end_ns);
                    }

                    const std::lock_guard<std::mutex> lock(mutex_);
                    span_ns_ = last > first ? last - first : 0;
                }

                /** @brief Returns the span of the last run in nanoseconds and forgets it. */
                [[nodiscard]] std::uint64_t take_span_ns()
                {
                    const std::lock_guard<std::mutex> lock(mutex_);
                    const std::uint64_t span = span_ns_;
                    span_ns_ = 0;
                    return span;
                }

            private:
                /// @brief Guards span_ns_, which a runtime thread writes.
                std::mutex mutex_;

                /// @brief The span of the last run, in nanoseconds.
                std::uint64_t span_ns_ = 0;
        };
    } // namespace Ceilings
} // namespace SushiBLAS
