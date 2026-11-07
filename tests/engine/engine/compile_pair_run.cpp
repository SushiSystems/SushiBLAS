/**************************************************************************/
/* compile_pair_run.cpp                                                   */
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
 * @file compile_pair_run.cpp
 * @brief Checks at compile time that a TaskSink body takes a PairRun and that a run is never built
 *        without a dispatcher; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/engine/pair_run.hpp>
#include <SushiBLAS/engine/task_sink.hpp>

namespace SushiBLAS
{
    static_assert(std::is_invocable_v<TaskSink::PairBody, PairRun&>,
                  "a recorded body is called with the run it reads and finishes");
    static_assert(!std::is_default_constructible_v<PairRun>,
                  "a run always has a dispatcher and its dependencies");
} // namespace SushiBLAS
