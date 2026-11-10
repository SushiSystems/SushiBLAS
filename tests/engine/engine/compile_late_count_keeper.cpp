/**************************************************************************/
/* compile_late_count_keeper.cpp                                          */
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
 * @file compile_late_count_keeper.cpp
 * @brief Checks at compile time that the runtime hold and the host keeper are adapters of LateCountKeeper; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/engine/host_late_count_keeper.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/late_count_keeper.hpp>

namespace SushiBLAS
{
    static_assert(std::is_base_of_v<LateCountKeeper, LateCountHold>, "the hold over a runtime is an adapter of the port");
    static_assert(std::is_base_of_v<LateCountKeeper, HostLateCountKeeper>, "the host keeper is an adapter of the port");
    static_assert(!std::is_default_constructible_v<LateCountKeeper>, "the port is never built alone");
} // namespace SushiBLAS
