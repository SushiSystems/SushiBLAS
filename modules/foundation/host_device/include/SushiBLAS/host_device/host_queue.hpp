/**************************************************************************/
/* host_queue.hpp                                                         */
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
 * @file host_queue.hpp
 * @brief Declares host_queue, the in-order queue on the first CPU device SYCL reports.
 * @author Mustafa Garip
 */

#pragma once

#include <optional>
#include <vector>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Detail
        {
            /** @brief Returns an in-order queue on the first of @p devices, or nothing when the list is empty. */
            [[nodiscard]] inline std::optional<sycl::queue> make_host_queue(
                const std::vector<sycl::device>& devices)
            {
                if (devices.empty())
                    return std::nullopt;
                return sycl::queue(devices.front(), sycl::property::queue::in_order());
            }

            /** @brief Returns an in-order queue on the first CPU device SYCL reports, or nothing without one. */
            [[nodiscard]] inline std::optional<sycl::queue> make_host_queue()
            {
                return make_host_queue(sycl::device::get_devices(sycl::info::device_type::cpu));
            }
        } // namespace Detail

        /**
         * @brief Returns the in-order queue on the host CPU device, or nullptr when there is none.
         * @return The same queue on every call; it lives until the program ends.
         */
        [[nodiscard]] inline sycl::queue* host_queue()
        {
            static std::optional<sycl::queue> queue = Detail::make_host_queue();
            return queue ? &*queue : nullptr;
        }
    } // namespace Test
} // namespace SushiBLAS
