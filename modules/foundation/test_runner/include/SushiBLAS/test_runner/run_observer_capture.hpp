/**************************************************************************/
/* run_observer_capture.hpp                                               */
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
 * @file run_observer_capture.hpp
 * @brief Declares RunObserverCapture, which keeps the dispatch records of the runs of a runtime for a test.
 * @author Mustafa Garip
 * @see modules/foundation/test_runner/README.md
 */

#pragma once

#include <mutex>
#include <memory>
#include <vector>
#include <cstddef>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/report/run_observer.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Observes a runtime while it lives and keeps one record per dispatch of every run.
         *
         * Adding the observer turns device timing on for the runtime's graphs, so the records
         * carry the widths a dispatch launched and did work over, the node it belongs to and
         * its times.
         * @see modules/foundation/test_runner/README.md
         */
        class RunObserverCapture final
        {
            public:
                /** @brief Starts observing @p runtime; the runtime must outlive this capture. */
                explicit RunObserverCapture(const SushiRuntime::Runtime& runtime)
                    : observer_(std::make_shared<Observer>()),
                      handle_(runtime.add_run_observer(observer_))
                {

                }

                RunObserverCapture(const RunObserverCapture&) = delete;
                RunObserverCapture& operator=(const RunObserverCapture&) = delete;
                RunObserverCapture(RunObserverCapture&&) = delete;
                RunObserverCapture& operator=(RunObserverCapture&&) = delete;
                ~RunObserverCapture() = default;

                /** @brief Returns the records of every run since construction or the last clear(), in arrival order. */
                [[nodiscard]] std::vector<SushiRuntime::NodeEventRecord> records() const
                {
                    return observer_->records();
                }

                /** @brief Returns how many runs the observer was told about since construction or the last clear(). */
                [[nodiscard]] std::size_t run_count() const
                {
                    return observer_->run_count();
                }

                /** @brief Forgets every record and run. */
                void clear()
                {
                    observer_->clear();
                }

            private:
                /** @brief Receives each finished run and appends its records. */
                class Observer final : public SushiRuntime::RunObserver
                {
                    public:
                        /** @brief Asks for device timing, which is what fills the widths. */
                        [[nodiscard]] bool wants_device_timing() const noexcept override
                        {
                            return true;
                        }

                        /** @brief Appends the records of one run. */
                        void on_run(const SushiRuntime::RunObservation& observation) noexcept override
                        {
                            const std::lock_guard<std::mutex> lock(mutex_);
                            events_.insert(events_.end(), observation.events.begin(),
                                           observation.events.end());
                            ++runs_;
                        }

                        /** @brief Returns a copy of the records held. */
                        [[nodiscard]] std::vector<SushiRuntime::NodeEventRecord> records() const
                        {
                            const std::lock_guard<std::mutex> lock(mutex_);
                            return events_;
                        }

                        /** @brief Returns how many runs were observed. */
                        [[nodiscard]] std::size_t run_count() const
                        {
                            const std::lock_guard<std::mutex> lock(mutex_);
                            return runs_;
                        }

                        /** @brief Forgets what was held. */
                        void clear()
                        {
                            const std::lock_guard<std::mutex> lock(mutex_);
                            events_.clear();
                            runs_ = 0;
                        }

                    private:
                        /// @brief Guards events_ and runs_, which a runtime thread fills.
                        mutable std::mutex mutex_;

                        /// @brief The records of every observed run.
                        std::vector<SushiRuntime::NodeEventRecord> events_;

                        /// @brief How many runs were observed.
                        std::size_t runs_ = 0;
                };

                /// @brief The observer the runtime holds a share of.
                std::shared_ptr<Observer> observer_;

                /// @brief Removes the observer when this capture goes.
                SushiRuntime::RunObserverHandle handle_;
        };
    } // namespace Test
} // namespace SushiBLAS
