/**************************************************************************/
/* local_tile.hpp                                                         */
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
 * @file local_tile.hpp
 * @brief Declares the local-memory demand of a tiled kernel and its record-time check.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/tensor_profile.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Returns the local tile a SYCL kernel stages for dtype.
         * @tparam Tile A type whose static `demand<T>()` returns a Device::LocalTileDemand.
         * @return An empty demand for an integer dtype.
         */
        template<typename Tile>
        [[nodiscard]] constexpr Device::LocalTileDemand local_tile_demand(
            Core::DataType dtype) noexcept
        {
            switch (dtype)
            {
                case Core::DataType::Half:
                    return Tile::template demand<sycl::half>();
                case Core::DataType::Float32:
                    return Tile::template demand<float>();
                case Core::DataType::Float64:
                    return Tile::template demand<double>();
                case Core::DataType::Complex32:
                    return Tile::template demand<std::complex<float>>();
                case Core::DataType::Complex64:
                    return Tile::template demand<std::complex<double>>();
                case Core::DataType::Int32:
                case Core::DataType::Int64:
                case Core::DataType::UInt32:
                    break;
            }
            return Device::LocalTileDemand{};
        }

        /**
         * @brief Returns whether the SYCL kernel of a tile may run on a device of profile.
         * @tparam Tile A type whose static `demand<T>()` returns a Device::LocalTileDemand.
         * @return True under ForceReference, which never runs that kernel, and when the device
         *         hosts the tile.
         */
        template<typename Tile>
        [[nodiscard]] bool tile_is_accepted(ExecutionMode mode, Core::DataType dtype,
                                            const Device::Profile& profile) noexcept
        {
            return mode == ExecutionMode::ForceReference ||
                   Device::hosts_local_tile(profile, local_tile_demand<Tile>(dtype));
        }

        /**
         * @brief Throws unless tile_is_accepted holds for the device of the given profile.
         * @throws UnsupportedExecution naming the demand and what the device offers.
         * @see modules/operations/blas/README.md
         */
        template<typename Tile>
        void require_local_tile(ExecutionMode mode, Core::DataType dtype,
                                const Device::Profile& profile, const char* op)
        {
            if (tile_is_accepted<Tile>(mode, dtype, profile))
                return;

            const Device::LocalTileDemand demand = local_tile_demand<Tile>(dtype);
            Dispatch::Detail::throw_execution_error(
                op,
                "the SYCL kernel stages {} B of local memory in {}-item work-groups; the device "
                "offers {} B and {}",
                demand.local_bytes, demand.work_group_items, profile.local_mem_bytes,
                profile.max_work_group_size);
        }

        /**
         * @brief Throws unless the device of C hosts the tile the SYCL kernel of op stages.
         * @tparam Tile A type whose static `demand<T>()` returns a Device::LocalTileDemand.
         * @throws UnsupportedExecution outside ForceReference, which never runs that kernel.
         * @see modules/operations/blas/README.md
         */
        template<typename Tile>
        void require_local_tile(const TaskRecorder& recorder, const Tensor& C, const char* op)
        {
            if (!C.storage)
                return;

            require_local_tile<Tile>(recorder.mode(), C.dtype, Dispatch::profile_of(C), op);
        }
    } // namespace Internal
} // namespace SushiBLAS
