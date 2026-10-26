/**************************************************************************/
/* profile_session.hpp                                                    */
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
 * @file profile_session.hpp
 * @brief Declares ProfileSession, which records a benchmark program under SushiProfiler when asked to.
 * @author Mustafa Garip
 * @see modules/operations/benchmark_harness/README.md
 */

#pragma once

#include <memory>
#include <string>
#include <cstdio>
#include <string_view>
#include <SushiBLAS/benchmark_harness/environment_value.hpp>
#include <SushiBLAS/profiling/profiler_parts.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/profiling/register_profiler_parts.hpp>
#include <SushiProfiler/json/json_value.hpp>
#include <SushiProfiler/session/session.hpp>

namespace SushiBLAS
{
    namespace Bench
    {
        /** @brief Names the environment variable that holds the folder a benchmark's record is written to. */
        inline constexpr const char* K_PROFILE_FOLDER_VARIABLE = "SB_BENCH_PROFILE";

        /** @brief Names the timeline each benchmark iteration marks one period of. */
        inline constexpr const char* K_ITERATION_TIMELINE = "bench.iteration";

        /**
         * @brief Records one benchmark program as a SushiProfiler run, one period per iteration.
         *
         * Bench::print_environment starts it when SB_BENCH_PROFILE names a folder; it ends with
         * the process and writes `<scenario>.sprec.json` there. Without the variable nothing is
         * recorded.
         * @see modules/operations/benchmark_harness/README.md
         */
        class ProfileSession final
        {
            public:
                /**
                 * @brief Starts a session over @p runtime when the folder variable is set.
                 * @param scenario The program's name; the record is `<scenario>.sprec.json`.
                 * @return The session, or null when the variable is unset, empty or the session did not start.
                 */
                [[nodiscard]] static ProfileSession* start_from_environment(SushiRuntime::Runtime& runtime,
                                                                            std::string_view scenario)
                {
                    static std::unique_ptr<ProfileSession> instance;
                    if (instance != nullptr)
                        return instance.get();

                    const std::string folder = environment_value(K_PROFILE_FOLDER_VARIABLE);
                    if (folder.empty())
                        return nullptr;

                    std::unique_ptr<ProfileSession> started(
                        new ProfileSession(runtime, scenario, folder));
                    if (!started->is_active())
                        return nullptr;
                    instance = std::move(started);
                    return instance.get();
                }

                /** @brief Returns the running session, or null when none was started. */
                [[nodiscard]] static ProfileSession* current() noexcept
                {
                    return current_;
                }

                ProfileSession(const ProfileSession&) = delete;
                ProfileSession& operator=(const ProfileSession&) = delete;
                ProfileSession(ProfileSession&&) = delete;
                ProfileSession& operator=(ProfileSession&&) = delete;

                /** @brief Stops the session and writes the record. */
                ~ProfileSession()
                {
                    if (session_ == nullptr)
                        return;

                    current_ = nullptr;
                    SushiProfiler::SessionStop stopped = session_->stop();
                    const SushiProfiler::SessionReport report = session_->report(stopped.record);
                    for (const std::string& problem : report.problems)
                        std::fprintf(stderr, "profile: %s\n", problem.c_str());
                }

                /** @brief Returns whether the session records. */
                [[nodiscard]] bool is_active() const noexcept
                {
                    return session_ != nullptr;
                }

                /** @brief Opens the next period: one benchmark iteration begins. */
                void mark_iteration()
                {
                    if (timeline_ != nullptr)
                        timeline_->mark();
                }

                /** @brief Opens the next period of the running session, if there is one. */
                static void mark_current_iteration()
                {
                    if (current_ != nullptr)
                        current_->mark_iteration();
                }

            private:
                ProfileSession(SushiRuntime::Runtime& runtime, std::string_view scenario,
                               const std::string& folder)
                {
                    runtime_parts_ = std::make_unique<SushiRuntime::RuntimeProfilerParts>(registry_, runtime);

                    blas_parts_.reset(new Profiling::SushiBLASProfilerParts(
                        Profiling::register_profiler_parts(registry_, runtime, folder)));

                    SushiProfiler::ProfilerConfig config;
                    config.timeline = K_ITERATION_TIMELINE;
                    config.window.warmup = 2U;
                    config.window.active = 10U;
                    config.window.repeat = 0U;
                    config.sources = {SushiProfiler::NamedSettings{"sushiruntime.device",
                                                                  SushiProfiler::JsonValue::object()}};
                    config.analyses = {"roofline", "top_spans"};
                    SushiProfiler::JsonValue record_settings = SushiProfiler::JsonValue::object();
                    record_settings.set("path", SushiProfiler::JsonValue::string(std::string(scenario) +
                                                                                 ".sprec.json"));
                    config.writers = {SushiProfiler::NamedSettings{"record", record_settings}};
                    config.output = folder;

                    SushiProfiler::SessionStart started = SushiProfiler::Session::start(config, registry_);
                    if (!started.ok())
                    {
                        std::fprintf(stderr, "profile: the session did not start: %s\n", started.error.c_str());
                        return;
                    }
                    session_ = std::move(started.session);
                    timeline_ = &session_->timeline(K_ITERATION_TIMELINE);
                    current_ = this;
                }

                /// @brief The running session, or null; one program has at most one.
                static inline ProfileSession* current_ = nullptr;

                /// @brief Holds what the parts below register.
                SushiProfiler::Registry registry_;

                /// @brief The runtime's source and contributor.
                std::unique_ptr<SushiRuntime::RuntimeProfilerParts> runtime_parts_;

                /// @brief SushiBLAS's contributor.
                std::unique_ptr<Profiling::SushiBLASProfilerParts> blas_parts_;

                /// @brief The running session, or null when none started.
                std::unique_ptr<SushiProfiler::Session> session_;

                /// @brief The timeline each iteration marks; owned by session_.
                SushiProfiler::PeriodTimeline* timeline_ = nullptr;
        };
    } // namespace Bench
} // namespace SushiBLAS
