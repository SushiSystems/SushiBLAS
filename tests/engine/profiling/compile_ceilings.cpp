/**************************************************************************/
/* compile_ceilings.cpp                                                   */
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
 * @file compile_ceilings.cpp
 * @brief Checks at compile time that every default of ComputeCeiling and DeviceCeilings states nothing a reader would accept as measured, through the literal tables the two structs are built from; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <SushiBLAS/profiling/ceilings.hpp>

namespace SushiBLAS
{
    static_assert(Profiling::BandwidthCeiling{}.bytes_per_second == 0.0, "a bandwidth ceiling starts at zero");
    static_assert(Profiling::BandwidthCeiling{}.runs == 0U, "a bandwidth ceiling starts with no timed run");
    static_assert(Profiling::CeilingsMethod{}.runs == 0U, "a default method states no timed run");

    namespace
    {
        constexpr Profiling::ComputeCeilingDefaults COMPUTE = Profiling::K_COMPUTE_CEILING_DEFAULTS;
        constexpr Profiling::DeviceCeilingsDefaults DEVICE = Profiling::K_DEVICE_CEILINGS_DEFAULTS;
    } // namespace

    static_assert(COMPUTE.dtype.empty(), "a default compute ceiling is for no dtype");
    static_assert(COMPUTE.flops_per_second == 0.0, "a default compute ceiling states no rate");
    static_assert(COMPUTE.spread == 0.0, "a default compute ceiling states no spread");
    static_assert(COMPUTE.runs == 0U, "a default compute ceiling states no timed run");

    static_assert(DEVICE.device_name.empty(), "a default device has no name");
    static_assert(DEVICE.backend.empty(), "a default device has no backend");
    static_assert(DEVICE.driver.empty(), "a default device has no driver");
    static_assert(!DEVICE.smoke_only, "a default device is not a smoke run");
    static_assert(DEVICE.method.runs == 0U && DEVICE.method.warmup_runs == 0U, "a default device method states no run");
    static_assert(DEVICE.method.min_span_ms == 0.0 && DEVICE.method.spread_limit == 0.0, "a default device method states no limit");
    static_assert(DEVICE.compute_count == 0U, "a default device has no compute ceiling");
    static_assert(DEVICE.bandwidth.bytes_per_second == 0.0 && DEVICE.bandwidth.runs == 0U, "a default device states no bandwidth");
    static_assert(DEVICE.launch_floor_ns == 0.0, "a default device states no launch floor");
} // namespace SushiBLAS
