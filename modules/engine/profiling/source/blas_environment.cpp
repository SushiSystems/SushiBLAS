/**************************************************************************/
/* blas_environment.cpp                                                   */
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
 * @file blas_environment.cpp
 * @brief Defines BlasEnvironment, the contributor of SushiBLAS's facts and device ceilings.
 * @author Mustafa Garip
 * @see modules/engine/profiling/README.md
 */

#include <SushiBLAS/profiling/blas_environment.hpp>

#include <string>
#include <utility>
#include <memory>
#include <sstream>
#include <optional>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/launch/gemm_tile_pin.hpp>
#include <SushiBLAS/profile/profile_gate.hpp>
#include <SushiBLAS/profile/profile_keys.hpp>
#include <SushiBLAS/profiling/ceilings_file.hpp>
#include <SushiBLAS/profiling/folder_ceilings_source.hpp>
#include <SushiBLAS/profiling/profiling_log_category.hpp>
#include <SushiProfiler/record/device_span_keys.hpp>

namespace SushiBLAS
{
    namespace Profiling
    {
        namespace
        {
            namespace Keys = Internal::Profile;

            /** @brief Names what became of a device's ceilings, as the status fact states it. */
            enum class CeilingsState : std::uint8_t
            {
                Measured,
                SmokeOnly,
                Unmeasured,
                Refused
            };

            /** @brief Returns the text of a status fact. */
            const char* state_text(CeilingsState state) noexcept
            {
                switch (state)
                {
                    case CeilingsState::Measured:
                        return "measured";
                    case CeilingsState::SmokeOnly:
                        return "smoke_only";
                    case CeilingsState::Unmeasured:
                        return "unmeasured";
                    case CeilingsState::Refused:
                        return "refused";
                }
                return "unmeasured";
            }

            /** @brief Returns a number as the text a ceiling fact holds: a decimal that round-trips. */
            std::string number_text(double value)
            {
                std::ostringstream stream;
                stream.precision(17);
                stream << value;
                return stream.str();
            }

            /** @brief Sets the facts of one device's ceilings, and returns what became of them. */
            CeilingsState state_ceilings(const DeviceFacts& device, const CeilingsSource& source,
                                         std::vector<SushiProfiler::Fact>& facts)
            {
                const std::string file_name = ceilings_file_name(device.device_name);
                const std::optional<std::string> text = source.read(file_name);
                if (!text)
                {
                    SB_LOG_INFO(K_PROFILING_LOG_CATEGORY,
                                "no ceilings file {} for device '{}'; its spans are stated without "
                                "a ceiling. Run device_ceilings to measure one.",
                                source.location(file_name), device.device_name);
                    return CeilingsState::Unmeasured;
                }

                const CeilingsReadResult read = read_ceilings(*text);
                if (!read.ok())
                {
                    SB_LOG_WARN(K_PROFILING_LOG_CATEGORY,
                                "ceilings file {} refused: {}; device '{}' is stated without a "
                                "ceiling.",
                                source.location(file_name), read.message, device.device_name);
                    return CeilingsState::Refused;
                }

                for (const ComputeCeiling& compute : read.ceilings.compute)
                {
                    SushiProfiler::set_fact(
                        facts, SushiProfiler::DeviceSpanKeys::peak_flops_fact(device.track,
                                                                              compute.dtype),
                        number_text(compute.flops_per_second));
                }
                SushiProfiler::set_fact(
                    facts, SushiProfiler::DeviceSpanKeys::peak_bytes_per_s_fact(device.track),
                    number_text(read.ceilings.bandwidth.bytes_per_second));
                return read.ceilings.smoke_only ? CeilingsState::SmokeOnly
                                                : CeilingsState::Measured;
            }
        } // namespace

        BlasEnvironment::BlasEnvironment(std::vector<DeviceFacts> devices,
                                         std::string ceilings_folder)
            : BlasEnvironment(std::move(devices),
                              std::make_shared<const FolderCeilingsSource>(std::move(ceilings_folder)))
        {

        }

        BlasEnvironment::BlasEnvironment(std::vector<DeviceFacts> devices,
                                         std::shared_ptr<const CeilingsSource> ceilings)
            : devices_(std::move(devices)), ceilings_(std::move(ceilings))
        {

        }

        void BlasEnvironment::contribute(SushiProfiler::RecordFacts& facts) const
        {
            SushiProfiler::set_fact(facts.build, std::string(Keys::K_FACT_VERSION),
                                    SUSHIBLAS_VERSION_STRING);
            SushiProfiler::set_fact(facts.build, std::string(Keys::K_FACT_PROFILE_LEVEL),
                                    std::to_string(Keys::K_COMPILED_LEVEL));

            SushiProfiler::set_fact(facts.environment, std::string(Keys::K_FACT_DEFAULT_MODE),
                                    "auto");
            const std::string& pin = Device::gemm_tile_pin();
            SushiProfiler::set_fact(facts.environment, std::string(Keys::K_FACT_TILE_PIN),
                                    pin.empty() ? std::string("unpinned") : pin);

            for (const DeviceFacts& device : devices_)
            {
                const CeilingsState state =
                    state_ceilings(device, *ceilings_, facts.environment);
                SushiProfiler::set_fact(
                    facts.environment,
                    std::string(Keys::K_FACT_CEILINGS_PREFIX) + device.track, state_text(state));
            }
        }
    } // namespace Profiling
} // namespace SushiBLAS
