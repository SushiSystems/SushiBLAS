/**************************************************************************/
/* compile_sparse.cpp                                                     */
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
 * @file compile_sparse.cpp
 * @brief Checks at compile time that SparseBLAS is built from a recorder and only from one; the build compiles it and no test runner lists it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/blas/sparse.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_constructible_v<SparseBLAS, TaskRecorder&>,
                      "SparseBLAS is built over a recorder.");
        static_assert(!std::is_convertible_v<TaskRecorder&, SparseBLAS>,
                      "A recorder does not turn into SparseBLAS without being named.");
        static_assert(!std::is_default_constructible_v<SparseBLAS>,
                      "SparseBLAS has no recorder-free form.");
    } // namespace Test
} // namespace SushiBLAS
