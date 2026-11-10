/**************************************************************************/
/* ceilings.hpp                                                           */
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
 * @file ceilings.hpp
 * @brief Declares DeviceCeilings, what a measurement run states about the limits of one device.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace SushiBLAS
{
    namespace Profiling
    {
        /**
         * @brief Holds the default of every field of a ComputeCeiling.
         * @note A literal type, so a static_assert can check what a default ceiling states. ComputeCeiling holds a
         *       std::string, which is not literal in C++17, and is built from this table.
         */
        struct ComputeCeilingDefaults
        {
            std::string_view dtype = "";     /**< No class: a default ceiling is for no dtype. */
            double flops_per_second = 0.0;   /**< No rate. */
            double spread = 0.0;             /**< No spread. */
            std::uint32_t runs = 0;          /**< No timed run. */
        };

        /** @brief The defaults a ComputeCeiling starts from. */
        inline constexpr ComputeCeilingDefaults K_COMPUTE_CEILING_DEFAULTS{};

        /** @brief Holds the compute ceiling a device reached in one element type. */
        struct ComputeCeiling
        {
            std::string dtype{K_COMPUTE_CEILING_DEFAULTS.dtype};                    /**< The class the ceiling is for: "f16", "f32" or "f64". */
            double flops_per_second = K_COMPUTE_CEILING_DEFAULTS.flops_per_second;  /**< The ceiling, in floating-point operations per second. */
            double spread = K_COMPUTE_CEILING_DEFAULTS.spread;                      /**< (maximum - minimum) / median over the timed runs. */
            std::uint32_t runs = K_COMPUTE_CEILING_DEFAULTS.runs;                   /**< How many timed runs the median is of. */
        };

        /** @brief Holds how the ceilings were measured. */
        struct CeilingsMethod
        {
            std::uint32_t warmup_runs = 0;   /**< Runs made and discarded before timing. */
            std::uint32_t runs = 0;          /**< Timed runs per ceiling. */
            double min_span_ms = 0.0;        /**< The shortest device span a timed run was allowed to have. */
            double spread_limit = 0.0;       /**< The spread above which a ceiling was refused. */
        };

        /** @brief Holds the memory bandwidth a device reached. */
        struct BandwidthCeiling
        {
            double bytes_per_second = 0.0;   /**< Bytes read plus bytes written per second. */
            double spread = 0.0;             /**< (maximum - minimum) / median over the timed runs. */
            std::uint32_t runs = 0;          /**< How many timed runs the median is of. */
        };

        /**
         * @brief Holds the default of every field of a DeviceCeilings.
         * @note A literal type, so a static_assert can check what a default device states. DeviceCeilings holds
         *       strings and a vector, which are not literal in C++17, and is built from this table.
         */
        struct DeviceCeilingsDefaults
        {
            std::string_view device_name = "";         /**< No device name. */
            std::string_view backend = "";             /**< No backend. */
            std::string_view driver = "";              /**< No driver. */
            bool smoke_only = false;                   /**< Not a smoke run. */
            CeilingsMethod method{};                   /**< A method that states no run. */
            std::size_t compute_count = 0;             /**< No compute ceiling. */
            BandwidthCeiling bandwidth{};              /**< A bandwidth that states no run. */
            double launch_floor_ns = 0.0;              /**< No launch floor. */
        };

        /** @brief The defaults a DeviceCeilings starts from. */
        inline constexpr DeviceCeilingsDefaults K_DEVICE_CEILINGS_DEFAULTS{};

        /**
         * @brief Holds the measured limits of one device: compute per dtype, bandwidth and launch floor.
         * @see docs/reference/CEILINGS_FORMAT.md
         */
        struct DeviceCeilings
        {
            std::string device_name{K_DEVICE_CEILINGS_DEFAULTS.device_name};        /**< The name the device reports. */
            std::string backend{K_DEVICE_CEILINGS_DEFAULTS.backend};                /**< The backend that drove it. */
            std::string driver{K_DEVICE_CEILINGS_DEFAULTS.driver};                  /**< The driver version the device reports. */
            bool smoke_only = K_DEVICE_CEILINGS_DEFAULTS.smoke_only;                /**< True when the numbers only prove the program runs, as on a CPU. */
            CeilingsMethod method = K_DEVICE_CEILINGS_DEFAULTS.method;              /**< How the numbers were measured. */
            std::vector<ComputeCeiling> compute;                                    /**< One ceiling per dtype the device supports. */
            BandwidthCeiling bandwidth = K_DEVICE_CEILINGS_DEFAULTS.bandwidth;      /**< The memory ceiling. */
            double launch_floor_ns = K_DEVICE_CEILINGS_DEFAULTS.launch_floor_ns;    /**< The device time of an empty kernel, in nanoseconds. */
        };
    } // namespace Profiling
} // namespace SushiBLAS
