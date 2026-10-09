/**************************************************************************/
/* task_handle.hpp                                                        */
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
 * @file task_handle.hpp
 * @brief Declares TaskHandle, the value a record call returns for the task it recorded.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    class TaskRecorder;

    /**
     * @brief Identifies one task a TaskRecorder recorded, or no task.
     * @see modules/engine/engine/README.md
     */
    class TaskHandle final
    {
        public:
            /** @brief Builds the handle of no task, which a call that records nothing returns. */
            constexpr TaskHandle() noexcept = default;

            /** @brief Reports whether the handle names a recorded task. */
            [[nodiscard]] constexpr bool is_valid() const noexcept
            {
                return recorder_ != 0;
            }

            /** @brief Reports whether two handles name the same record call, or both name none. */
            [[nodiscard]] friend constexpr bool operator==(TaskHandle a, TaskHandle b) noexcept
            {
                return a.recorder_ == b.recorder_ && a.registration_ == b.registration_;
            }

            /** @brief Reports whether two handles name different record calls. */
            [[nodiscard]] friend constexpr bool operator!=(TaskHandle a, TaskHandle b) noexcept
            {
                return !(a == b);
            }

        private:
            friend class TaskRecorder;

            /** @brief Builds the handle of the task a recorder inserted under a registration index. */
            constexpr TaskHandle(std::uint64_t recorder, std::uint32_t registration) noexcept
                : recorder_(recorder), registration_(registration)
            {

            }

            /// @brief Identity of the recorder that recorded the task; 0 names no task.
            std::uint64_t recorder_ = 0;

            /// @brief Registration index the recorder stamped on the task's graph node.
            std::uint32_t registration_ = 0;
    };
} // namespace SushiBLAS
