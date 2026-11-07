/**************************************************************************/
/* tensor_profile.hpp                                                     */
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


/**
 * @file tensor_profile.hpp
 * @brief Declares profile_of(), the device Profile an operation sizes its work for from a tensor.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Returns the Profile of the device that holds a tensor's storage, or the host
         *        Profile when a host engine made the storage.
         * @throws std::runtime_error if the tensor has no storage.
         * @see modules/engine/engine/README.md
         */
        [[nodiscard]] inline const Device::Profile& profile_of(const Tensor& t)
        {
            SB_THROW_IF(!t.storage, "Tensor has no valid storage/context");
            if (t.storage->is_host())
                return Device::host_profile();
            return Device::profile_for(t.device());
        }
    } // namespace Dispatch
} // namespace SushiBLAS
