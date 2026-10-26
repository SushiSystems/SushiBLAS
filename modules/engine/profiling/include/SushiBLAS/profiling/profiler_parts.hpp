/**************************************************************************/
/* profiler_parts.hpp                                                     */
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
 * @file profiler_parts.hpp
 * @brief Declares SushiBLASProfilerParts and register_profiler_parts, the one way SushiBLAS joins a profiler registry.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#pragma once

#include <string>
#include <vector>
#include <string_view>
#include <SushiBLAS/profiling/blas_environment.hpp>
#include <SushiProfiler/registry/registry.hpp>

namespace SushiRuntime
{
    class Runtime;
} // namespace SushiRuntime

namespace SushiBLAS
{
    namespace Profiling
    {
        /**
         * @brief Owns what SushiBLAS adds to a SushiProfiler registry: its environment contributor.
         *
         * The registry holds the contributor by reference, so the caller keeps this object until
         * every session started with the registry has stopped. It is neither copyable nor
         * movable; register_profiler_parts returns it as a prvalue.
         * @see modules/engine/profiling/README.md
         */
        class SushiBLASProfilerParts final
        {
            public:
                /**
                 * @brief Adds the contributor to @p registry.
                 * @throws std::invalid_argument when @p registry already holds a SushiBLAS contributor.
                 */
                SushiBLASProfilerParts(SushiProfiler::Registry& registry,
                                       std::vector<DeviceFacts> devices,
                                       std::string ceilings_folder);

                SushiBLASProfilerParts(const SushiBLASProfilerParts&) = delete;
                SushiBLASProfilerParts& operator=(const SushiBLASProfilerParts&) = delete;
                SushiBLASProfilerParts(SushiBLASProfilerParts&&) = delete;
                SushiBLASProfilerParts& operator=(SushiBLASProfilerParts&&) = delete;
                ~SushiBLASProfilerParts() = default;

                /** @brief Returns the contributor the registry holds. */
                [[nodiscard]] const BlasEnvironment& environment() const noexcept
                {
                    return environment_;
                }

            private:
                /// @brief The contributor the registry refers to.
                BlasEnvironment environment_;
        };

        /**
         * @brief Adds SushiBLAS's parts to @p registry for the devices given.
         * @param ceilings_folder The folder that holds the `<device>.ceilings.json` files.
         * @throws std::invalid_argument when @p registry already holds a SushiBLAS contributor.
         */
        [[nodiscard]] SushiBLASProfilerParts register_profiler_parts(
            SushiProfiler::Registry& registry, std::vector<DeviceFacts> devices,
            std::string_view ceilings_folder);

        /**
         * @brief Adds SushiBLAS's parts to @p registry for every device of @p runtime, named as the
         *        runtime's own device source names its tracks.
         * @throws std::invalid_argument when @p registry already holds a SushiBLAS contributor.
         */
        [[nodiscard]] SushiBLASProfilerParts register_profiler_parts(
            SushiProfiler::Registry& registry, const SushiRuntime::Runtime& runtime,
            std::string_view ceilings_folder);
    } // namespace Profiling
} // namespace SushiBLAS
