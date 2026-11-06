/**************************************************************************/
/* compile_combiners.cpp                                                  */
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
 * @file compile_combiners.cpp
 * @brief Checks at compile time that every combiner is trivially copyable; compiled, never run.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <type_traits>

#include <SushiBLAS/graph_primitives/combiners.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_trivially_copyable<GraphPrimitives::Sum<float>>::value,
                      "A device kernel captures the combiner by value.");
        static_assert(std::is_trivially_copyable<GraphPrimitives::Product<float>>::value,
                      "A device kernel captures the combiner by value.");
        static_assert(std::is_trivially_copyable<GraphPrimitives::Minimum<float>>::value,
                      "A device kernel captures the combiner by value.");
        static_assert(std::is_trivially_copyable<GraphPrimitives::Maximum<float>>::value,
                      "A device kernel captures the combiner by value.");
    } // namespace Test
} // namespace SushiBLAS
