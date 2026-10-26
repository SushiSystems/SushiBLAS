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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
#include <cstdint>

namespace SushiBLAS
{
    namespace Profiling
    {
        /** @brief Holds the compute ceiling a device reached in one element type. */
        struct ComputeCeiling
        {
            std::string dtype;               /**< The class the ceiling is for: "f16", "f32" or "f64". */
            double flops_per_second = 0.0;   /**< The ceiling, in floating-point operations per second. */
            double spread = 0.0;             /**< (maximum - minimum) / median over the timed runs. */
            std::uint32_t runs = 0;          /**< How many timed runs the median is of. */
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
         * @brief Holds the measured limits of one device: compute per dtype, bandwidth and launch floor.
         * @see docs/reference/CEILINGS_FORMAT.md
         */
        struct DeviceCeilings
        {
            std::string device_name;                   /**< The name the device reports. */
            std::string backend;                       /**< The backend that drove it. */
            std::string driver;                        /**< The driver version the device reports. */
            bool smoke_only = false;                   /**< True when the numbers only prove the program runs, as on a CPU. */
            CeilingsMethod method;                     /**< How the numbers were measured. */
            std::vector<ComputeCeiling> compute;       /**< One ceiling per dtype the device supports. */
            BandwidthCeiling bandwidth;                /**< The memory ceiling. */
            double launch_floor_ns = 0.0;              /**< The device time of an empty kernel, in nanoseconds. */
        };
    } // namespace Profiling
} // namespace SushiBLAS
