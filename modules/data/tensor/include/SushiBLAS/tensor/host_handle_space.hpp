/**************************************************************************/
/* host_handle_space.hpp                                                  */
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
 * @file host_handle_space.hpp
 * @brief Declares the one LocalHandleSpace that names every HostBlock of the library.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiRuntime/handle/local_handle_space.hpp>

namespace SushiBLAS
{
    /**
     * @brief The space id of the handles host blocks carry, in the machine field.
     *
     * Another SushiBLAS space needs another id; the host double of graph_primitives takes 2.
     */
    inline constexpr std::uint16_t K_HOST_HANDLE_SPACE_ID = 1;

    /**
     * @brief Returns the space that mints the handle of every HostBlock in the process.
     *
     * The space is a function-local static, so one exists per process however many translation
     * units include this header. Minting takes no lock, and the slots count in mint order.
     * @see modules/data/tensor/README.md
     */
    [[nodiscard]] inline SushiRuntime::LocalHandleSpace& host_handle_space() noexcept
    {
        static SushiRuntime::LocalHandleSpace space(K_HOST_HANDLE_SPACE_ID);
        return space;
    }
} // namespace SushiBLAS
