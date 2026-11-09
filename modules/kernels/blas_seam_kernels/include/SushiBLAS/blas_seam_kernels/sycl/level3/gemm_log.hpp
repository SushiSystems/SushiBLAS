/**************************************************************************/
/* gemm_log.hpp                                                           */
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
 * @file gemm_log.hpp
 * @brief Declares the records the SYCL GEMM keeps once, and the sink they are written to.
 * @author Mustafa Garip
 */

#pragma once

#include <set>
#include <mutex>
#include <tuple>
#include <string>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/launch/gemm_tile_pin.hpp>
#include <SushiBLAS/blas_seam_kernels/blas_seam_kernels_log_category.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_choice.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Takes the finished text of one GEMM record.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                class GemmLogSink
                {
                    public:
                        virtual ~GemmLogSink() = default;

                        /** @brief Takes a record of a fallback. */
                        virtual void warn(const std::string& message) = 0;

                        /** @brief Takes a record of a state. */
                        virtual void info(const std::string& message) = 0;
                };

                /**
                 * @brief Writes the records to the process logger, in the log category of the module.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                class LoggerGemmLogSink final : public GemmLogSink
                {
                    public:
                        /** @brief Logs message as a WARN. */
                        void warn(const std::string& message) override
                        {
                            SB_LOG_WARN(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, "{}", message);
                        }

                        /** @brief Logs message as an INFO. */
                        void info(const std::string& message) override
                        {
                            SB_LOG_INFO(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, "{}", message);
                        }
                };

                /**
                 * @brief Writes each GEMM record once to a sink, and remembers which it has written.
                 *
                 * A training or inference loop reaches these on every GEMM of every step, so a
                 * record is kept to the first time. The memory of what was written belongs to this
                 * object, so a test that builds its own starts with none.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                class GemmFallbackLog final
                {
                    public:
                        /** @param sink Receives the records; it outlives this object. */
                        explicit GemmFallbackLog(GemmLogSink& sink) noexcept : sink_(sink)
                        {

                        }

                        GemmFallbackLog(const GemmFallbackLog&) = delete;
                        GemmFallbackLog& operator=(const GemmFallbackLog&) = delete;

                        /**
                         * @brief Writes one WARN for each device and local tile that does not fit it,
                         *        and no more.
                         */
                        void warn_no_local_fallback(const Device::Profile& profile,
                                                    std::size_t tile_bytes, std::size_t tile_items)
                        {
                            const std::lock_guard<std::mutex> lock(guard_);
                            if (!warned_.emplace(&profile, tile_bytes, tile_items).second)
                                return;

                            sink_.warn(SushiLog::format_text(
                                "SYCL GEMM: local tile wants {} B and {} work-items, device offers "
                                "{} B and {}; falling back to the no-local variant.",
                                tile_bytes, tile_items, profile.local_mem_bytes,
                                profile.max_work_group_size));
                        }

                        /**
                         * @brief Writes one INFO when the pin was applied: the pin read and the tile
                         *        it chose.
                         */
                        void state_applied_gemm_tile_pin(const GemmTileChoice& choice, int64_t m,
                                                         int64_t n, int64_t k)
                        {
                            if (choice.pin != GemmTilePinStatus::Applied || !choice.role)
                                return;

                            std::call_once(stated_, [&]()
                            {
                                sink_.info(SushiLog::format_text(
                                    "SYCL GEMM: {}={} selects the {} tile; first seen for {}x{}x{}",
                                    Device::K_GEMM_TILE_PIN_VARIABLE, Device::gemm_tile_pin(),
                                    to_string(*choice.role), m, n, k));
                            });
                        }

                        /**
                         * @brief Writes one WARN when the pin names no catalog shape or is past a
                         *        hard limit of the device, and no more.
                         */
                        void warn_rejected_gemm_tile_pin(GemmTilePinStatus status)
                        {
                            if (status != GemmTilePinStatus::Unknown && status != GemmTilePinStatus::Refused)
                                return;

                            std::call_once(rejected_, [&]()
                            {
                                sink_.warn(SushiLog::format_text(
                                    "SYCL GEMM: {}={} {}; the tile is chosen as if unpinned.",
                                    Device::K_GEMM_TILE_PIN_VARIABLE, Device::gemm_tile_pin(),
                                    status == GemmTilePinStatus::Unknown
                                        ? "names no catalog shape"
                                        : "is past a hard limit of the device"));
                            });
                        }

                    private:
                        /// @brief Receives each record.
                        GemmLogSink& sink_;

                        /// @brief Guards warned_.
                        std::mutex guard_;

                        /// @brief The device and tile pairs already written.
                        std::set<std::tuple<const Device::Profile*, std::size_t, std::size_t>> warned_;

                        /// @brief Set once the applied pin is written.
                        std::once_flag stated_;

                        /// @brief Set once the rejected pin is written.
                        std::once_flag rejected_;
                };

                /** @brief Returns the log of the process: once per process, to the process logger. */
                [[nodiscard]] inline GemmFallbackLog& process_gemm_fallback_log()
                {
                    static LoggerGemmLogSink sink;
                    static GemmFallbackLog log(sink);

                    return log;
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
