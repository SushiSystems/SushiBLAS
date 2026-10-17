/**************************************************************************/
/* task_description.hpp                                                   */
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
 * @file task_description.hpp
 * @brief Declares TaskDescription, what an operation states about the task it records.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/engine_log_category.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds the name, the operation and the scalar parameters of one recorded task.
     * @see modules/engine/engine/README.md
     */
    struct TaskDescription
    {
        /** @brief Holds how many parameter slots a task carries. */
        static constexpr std::size_t K_MAX_PARAMETERS = 8;

        /** @brief The label the task reports under; must outlive the record call. */
        const char* name = "unnamed_task";

        /** @brief The operation the task performs; a default id names none. */
        SushiRuntime::OperationID operation = SushiRuntime::OperationID();

        /** @brief The parameter slots; a slot never written holds 0. */
        std::uint64_t parameters[K_MAX_PARAMETERS] = {0};

        /**
         * @brief Writes the bytes of value to one slot.
         * @tparam T Trivially copyable and at most 64 bits wide.
         * @pre 0 <= index < K_MAX_PARAMETERS; a release build ignores an index outside it.
         */
        template<typename T>
        void set_parameter(int index, T value) noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>, "A parameter is copied byte by byte.");
            static_assert(sizeof(T) <= sizeof(std::uint64_t), "A parameter fits in one slot.");
            SB_ASSERT(K_ENGINE_LOG_CATEGORY, index >= 0 && index < static_cast<int>(K_MAX_PARAMETERS),
                      "A task parameter was written outside the slots of TaskDescription");
            if (index >= 0 && index < static_cast<int>(K_MAX_PARAMETERS))
                std::memcpy(&parameters[index], &value, sizeof(T));
        }

        /**
         * @brief Returns the bytes of one slot read as T.
         * @tparam T Trivially copyable and at most 64 bits wide.
         * @pre 0 <= index < K_MAX_PARAMETERS; a release build returns T{} for an index outside it.
         */
        template<typename T>
        [[nodiscard]] T parameter(int index) const noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>, "A parameter is copied byte by byte.");
            static_assert(sizeof(T) <= sizeof(std::uint64_t), "A parameter fits in one slot.");
            SB_ASSERT(K_ENGINE_LOG_CATEGORY, index >= 0 && index < static_cast<int>(K_MAX_PARAMETERS),
                      "A task parameter was read outside the slots of TaskDescription");
            T value{};
            if (index >= 0 && index < static_cast<int>(K_MAX_PARAMETERS))
                std::memcpy(&value, &parameters[index], sizeof(T));
            return value;
        }
    };
} // namespace SushiBLAS
