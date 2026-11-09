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
 * @brief Checks at compile time that a default bandwidth ceiling and method state nothing a reader would accept as measured; the strings and the vector of DeviceCeilings are not literal types; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <SushiBLAS/profiling/ceilings.hpp>

namespace SushiBLAS
{
    static_assert(Profiling::BandwidthCeiling{}.bytes_per_second == 0.0, "a bandwidth ceiling starts at zero");
    static_assert(Profiling::BandwidthCeiling{}.runs == 0U, "a bandwidth ceiling starts with no timed run");
    static_assert(Profiling::CeilingsMethod{}.runs == 0U, "a default method states no timed run");
} // namespace SushiBLAS
