/**************************************************************************/
/* profiler_parts.cpp                                                     */
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
 * @file profiler_parts.cpp
 * @brief Defines SushiBLASProfilerParts and the two register_profiler_parts overloads.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#include <SushiBLAS/profiling/profiler_parts.hpp>

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/profiling/device_track_name.hpp>

namespace SushiBLAS
{
    namespace Profiling
    {
        namespace
        {
            /** @brief Throws unless @p registry has no SushiBLAS contributor yet. */
            void require_unregistered(const SushiProfiler::Registry& registry)
            {
                for (const SushiProfiler::EnvironmentContributor* contributor :
                     registry.contributors())
                {
                    if (dynamic_cast<const BlasEnvironment*>(contributor) != nullptr)
                        throw std::invalid_argument(
                            "register_profiler_parts: the registry already holds the SushiBLAS "
                            "contributor; a registry takes one set of SushiBLAS parts");
                }
            }
        } // namespace

        SushiBLASProfilerParts::SushiBLASProfilerParts(SushiProfiler::Registry& registry,
                                                       std::vector<DeviceFacts> devices,
                                                       std::string ceilings_folder)
            : environment_(std::move(devices), std::move(ceilings_folder))
        {
            require_unregistered(registry);
            registry.add_contributor(environment_);
        }

        SushiBLASProfilerParts register_profiler_parts(SushiProfiler::Registry& registry,
                                                       std::vector<DeviceFacts> devices,
                                                       std::string_view ceilings_folder)
        {
            return SushiBLASProfilerParts(registry, std::move(devices),
                                          std::string(ceilings_folder));
        }

        SushiBLASProfilerParts register_profiler_parts(SushiProfiler::Registry& registry,
                                                       const SushiRuntime::Runtime& runtime,
                                                       std::string_view ceilings_folder)
        {
            const auto& descriptions = runtime.advanced().devices();
            std::vector<DeviceFacts> devices;
            devices.reserve(descriptions.size());
            for (std::size_t index = 0; index < descriptions.size(); ++index)
            {
                devices.push_back(DeviceFacts{
                    SushiRuntime::device_track_name(runtime, static_cast<std::uint32_t>(index)),
                    descriptions[index].name});
            }
            return register_profiler_parts(registry, std::move(devices), ceilings_folder);
        }
    } // namespace Profiling
} // namespace SushiBLAS
