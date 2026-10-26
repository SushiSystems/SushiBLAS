/**************************************************************************/
/* submitted_events.hpp                                                   */
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
 * @file submitted_events.hpp
 * @brief Declares SubmittedEvents, the events of every kernel one task body submitted.
 * @author Mustafa Garip
 * @see modules/foundation/launch/README.md
 */

#pragma once

#include <vector>
#include <utility>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Device
    {
        /**
         * @brief Holds the events of the kernels a body submitted, in submission order.
         *
         * A body that submits several kernels returns this type, so the runtime can time the
         * span from the first kernel's start to the last kernel's end. It converts to the last
         * event, so a caller that orders against one event takes it unchanged.
         * @see modules/foundation/launch/README.md
         */
        class SubmittedEvents final
        {
            public:
                /** @brief Holds no event. */
                SubmittedEvents() = default;

                /** @brief Holds the one event of a kernel that was submitted alone. */
                SubmittedEvents(sycl::event event)
                {
                    events_.push_back(std::move(event));
                }

                /** @brief Holds @p events, in the order the kernels were submitted. */
                explicit SubmittedEvents(std::vector<sycl::event> events) noexcept
                    : events_(std::move(events))
                {

                }

                /** @brief Adds the event of the kernel submitted next. */
                void append(sycl::event event)
                {
                    events_.push_back(std::move(event));
                }

                /** @brief Adds every event of @p later, which was submitted after these. */
                void append(const SubmittedEvents& later)
                {
                    events_.insert(events_.end(), later.events_.begin(), later.events_.end());
                }

                /** @brief Returns every event, in submission order. */
                [[nodiscard]] const std::vector<sycl::event>& events() const noexcept
                {
                    return events_;
                }

                /** @brief Returns whether no kernel was submitted. */
                [[nodiscard]] bool empty() const noexcept
                {
                    return events_.empty();
                }

                /** @brief Returns the event of the last kernel, or an empty event when none was submitted. */
                operator sycl::event() const
                {
                    return events_.empty() ? sycl::event() : events_.back();
                }

                /** @brief Blocks until every kernel has finished. */
                void wait() const
                {
                    sycl::event::wait(events_);
                }

                /** @brief Blocks until every kernel has finished and rethrows the asynchronous errors. */
                void wait_and_throw() const
                {
                    sycl::event::wait_and_throw(events_);
                }

            private:
                /// @brief The events of the submitted kernels, in order.
                std::vector<sycl::event> events_;
        };
    } // namespace Device
} // namespace SushiBLAS
