/**************************************************************************/
/* blas_environment.hpp                                                   */
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
 * @file blas_environment.hpp
 * @brief Declares BlasEnvironment, which adds SushiBLAS's facts and its devices' ceilings to a record.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#pragma once

#include <string>
#include <vector>
#include <SushiProfiler/environment/environment_contributor.hpp>
#include <SushiProfiler/environment/record_facts.hpp>

namespace SushiBLAS
{
    namespace Profiling
    {
        /** @brief Names one device the way the runtime names its track, and the way it names itself. */
        struct DeviceFacts
        {
            std::string track;       /**< The track name the runtime's device source puts the device's spans on. */
            std::string device_name; /**< The name the device reports; its ceilings file is named after it. */
        };

        /**
         * @brief States the version and settings of SushiBLAS and the ceilings of each device it was given.
         *
         * A device whose ceilings file is missing or refused is stated `unmeasured` or `refused`
         * and gets no ceiling fact, so no analysis judges it against a number nobody measured.
         * @see modules/engine/profiling/README.md
         */
        class BlasEnvironment final : public SushiProfiler::EnvironmentContributor
        {
            public:
                /**
                 * @param devices The devices a record's spans can name.
                 * @param ceilings_folder The folder that holds the `<device>.ceilings.json` files.
                 */
                BlasEnvironment(std::vector<DeviceFacts> devices, std::string ceilings_folder);

                /** @brief Sets the facts of SushiBLAS, then the ceiling facts of each device. */
                void contribute(SushiProfiler::RecordFacts& facts) const override;

            private:
                /// @brief The devices a record's spans can name.
                std::vector<DeviceFacts> devices_;

                /// @brief The folder the ceilings files are read from.
                std::string ceilings_folder_;
        };
    } // namespace Profiling
} // namespace SushiBLAS
