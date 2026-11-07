/**************************************************************************/
/* compile_shared_allocator.cpp                                           */
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
 * @file compile_shared_allocator.cpp
 * @brief Checks at compile time that SharedAllocator is an interface an adapter implements and a holder takes by reference.
 * @author Mustafa Garip
 */

#include <type_traits>

#include <SushiBLAS/host_device/shared_allocator.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        static_assert(std::is_abstract_v<SharedAllocator>, "A SharedAllocator is implemented by an adapter.");
        static_assert(!std::is_copy_constructible_v<SharedAllocator>, "A port is not copied.");
        static_assert(!std::is_move_constructible_v<SharedAllocator>, "A port is not moved.");
        static_assert(std::has_virtual_destructor_v<SharedAllocator>,
                      "An adapter is destroyed through the port.");
    } // namespace Test
} // namespace SushiBLAS
