/**************************************************************************/
/* device_policy.hpp                                                      */
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
 * @file device_policy.hpp
 * @brief Declares the device profile and the launch geometry a kernel dispatches on.
 * @author Mustafa Garip
 */

#pragma once

#include <deque>
#include <mutex>
#include <vector>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace SushiBLAS
{
    namespace Device
    {
        /**
         * @brief Holds the number of resident work-groups per compute unit a launch aims for.
         * @see modules/foundation/launch/README.md
         */
        inline constexpr std::size_t K_WAVES_PER_COMPUTE_UNIT = 8;

        /** @brief Work-group size a GPU launch aims for before clamping. */
        inline constexpr std::size_t K_GPU_WORK_GROUP_TARGET = 256;

        /**
         * @brief Work-group size a CPU launch aims for before clamping.
         *
         * Small on purpose. On a CPU backend a work-group is a chunk handed to one
         * OS thread and every barrier inside it is a real rendezvous, so wide
         * groups buy nothing and cost synchronisation.
         */
        inline constexpr std::size_t K_CPU_WORK_GROUP_TARGET = 64;

        /** @brief Width in bytes of the widest single memory transaction to aim for. */
        inline constexpr std::size_t K_VECTOR_TRANSACTION_BYTES = 16;

        /**
         * @brief Holds the first-level data cache assumed for one compute unit, in bytes.
         * @see modules/foundation/launch/README.md
         */
        inline constexpr std::size_t K_ASSUMED_INNER_CACHE_BYTES = 32 * 1024;

        /** @brief Holds the cache size taken for a device that reports none, in bytes. */
        inline constexpr std::size_t K_ASSUMED_CACHE_BYTES = 256 * 1024;

        /**
         * @brief Holds the byte budgets and thresholds a cache-blocked kernel sizes itself from.
         * @see modules/foundation/launch/README.md
         */
        struct CacheBlocking
        {
            /** @brief Bytes the innermost loop keeps resident; it sets the depth of a block. */
            std::size_t inner_bytes = K_ASSUMED_INNER_CACHE_BYTES / 8 * 7;

            /** @brief Bytes of the packed block the inner loops sweep again; it sets the rows. */
            std::size_t outer_bytes = K_ASSUMED_CACHE_BYTES / 4;

            /** @brief Bytes of a block walked once per sweep; it sets the columns and the tile. */
            std::size_t stream_bytes = K_ASSUMED_CACHE_BYTES / 2;

            /** @brief Operand bytes of one product below which a kernel does not pack. */
            std::size_t min_operand_bytes = K_ASSUMED_CACHE_BYTES;

            /** @brief Register blocks the output must span on each axis before a kernel packs. */
            std::size_t min_register_blocks = 4;

            /** @brief Work-items a blocked launch starts per compute unit. */
            std::size_t workers_per_compute_unit = 1;
        };

        /**
         * @brief Returns the blocking a device that reports cache_bytes of cache starts from.
         * @see modules/foundation/launch/README.md
         */
        [[nodiscard]] inline constexpr CacheBlocking
        cache_blocking_for(std::size_t cache_bytes) noexcept
        {
            CacheBlocking blocking;
            blocking.outer_bytes = cache_bytes / 4;
            blocking.stream_bytes = cache_bytes / 2;
            blocking.min_operand_bytes = cache_bytes;
            return blocking;
        }

        /**
         * @brief Holds the device properties a kernel is allowed to dispatch on.
         * @see modules/foundation/launch/README.md
         */
        struct Profile
        {
            /** @brief True when the SYCL runtime classifies the device as a CPU. */
            bool is_cpu = true;

            /**
             * @brief Holds whether scratchpad memory is physically separate storage.
             * @see modules/foundation/launch/README.md
             */
            bool has_dedicated_local_memory = false;

            /** @brief True when the device supports double precision natively. */
            bool supports_fp64 = false;

            /** @brief True when the device supports half precision natively. */
            bool supports_fp16 = false;

            /**
             * @brief Holds whether a CUDA device offers HALF x HALF -> FLOAT32 joint_matrix
             *        products of 16 x 16 x 16.
             * @see modules/foundation/launch/README.md
             */
            bool supports_half_matrix = false;

            /** @brief Bytes of work-group local memory available to one group. */
            std::size_t local_mem_bytes = 0;

            /** @brief Largest work-group the device will accept. */
            std::size_t max_work_group_size = 1;

            /**
             * @brief The sub-group width to assume, in work-items.
             *
             * 32 on NVIDIA, 64 on AMD's wave64 parts, one of {8,16,32} on Intel.
             * Kernels use it to keep work-group sizes a whole number of sub-groups
             * and to know how wide a barrier-free shuffle reduction can be.
             */
            std::size_t sub_group_size = 1;

            /** @brief Independent execution units: SMs, CUs, or hardware threads. */
            std::size_t compute_units = 1;

            /** @brief Bytes of global-memory cache the device reports, or K_ASSUMED_CACHE_BYTES. */
            std::size_t cache_bytes = K_ASSUMED_CACHE_BYTES;

            /** @brief The budgets a cache-blocked kernel sizes its blocks from. */
            CacheBlocking blocking{};
        };

        namespace Detail
        {
            /**
             * @brief Returns whether a CUDA device lists a HALF x HALF -> FLOAT32 joint_matrix
             *        combination of 16 x 16 x 16.
             * @see modules/foundation/launch/README.md
             */
            [[nodiscard]] inline bool offers_half_matrix(const sycl::device& d)
            {
                namespace Experimental = sycl::ext::oneapi::experimental;
                namespace Matrix = Experimental::matrix;
                using Matrix::matrix_type;

                // Only CUDA is asked: the HALF matrix kernel is compiled for NVPTX alone.
                if (d.get_backend() != sycl::backend::ext_oneapi_cuda)
                    return false;

                const std::vector<Matrix::combination> combinations =
                    d.get_info<Experimental::info::device::matrix_combinations>();
                for (const Matrix::combination& c : combinations)
                {
                    const bool types = c.atype == matrix_type::fp16 &&
                                       c.btype == matrix_type::fp16 &&
                                       c.ctype == matrix_type::fp32 &&
                                       c.dtype == matrix_type::fp32;
                    const bool shape = c.msize == 16 && c.nsize == 16 && c.ksize == 16;
                    if (types && shape)
                        return true;
                }
                return false;
            }

            /** @brief Queries every Profile field from a device. Not cached; call profile_for. */
            inline Profile probe(const sycl::device& d)
            {
                Profile p;
                p.is_cpu = d.is_cpu();
                p.supports_fp64 = d.has(sycl::aspect::fp64);
                p.supports_fp16 = d.has(sycl::aspect::fp16);
                p.supports_half_matrix = offers_half_matrix(d);
                p.local_mem_bytes = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::local_mem_size>());
                p.max_work_group_size = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::max_work_group_size>());
                p.compute_units = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::max_compute_units>());

                const bool declares_local =
                    d.get_info<sycl::info::device::local_mem_type>() ==
                    sycl::info::local_mem_type::local;
                p.has_dedicated_local_memory = declares_local && !p.is_cpu &&
                                               p.local_mem_bytes > 0;

                // The widest sub-group is kept; a zero width is guarded; see modules/foundation/launch/README.md.
                const std::vector<std::size_t> widths =
                    d.get_info<sycl::info::device::sub_group_sizes>();
                p.sub_group_size = widths.empty()
                    ? std::size_t{1}
                    : *std::max_element(widths.begin(), widths.end());
                if (p.sub_group_size == 0)
                    p.sub_group_size = 1;

                if (p.max_work_group_size == 0)
                    p.max_work_group_size = 1;
                if (p.compute_units == 0)
                    p.compute_units = 1;

                const std::size_t reported_cache = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::global_mem_cache_size>());
                p.cache_bytes = reported_cache ? reported_cache : K_ASSUMED_CACHE_BYTES;
                p.blocking = cache_blocking_for(p.cache_bytes);

                return p;
            }
        } // namespace Detail

        /**
         * @brief Returns the cached Profile of a device, probing it on first sight.
         * @return A reference valid for the rest of the process.
         * @see modules/foundation/launch/README.md
         */
        inline const Profile& profile_for(const sycl::device& d)
        {
            // The cache is a deque for stable references and is never freed; see modules/foundation/launch/README.md.
            using Entry = std::pair<sycl::device, Profile>;
            static std::deque<Entry>* cache = new std::deque<Entry>();
            static std::mutex guard;

            // A thread's last entry answers a repeated device without the lock; entries never move.
            thread_local const Entry* last = nullptr;
            if (last != nullptr && last->first == d)
                return last->second;

            const std::lock_guard<std::mutex> lock(guard);
            for (const Entry& entry : *cache)
                if (entry.first == d)
                {
                    last = &entry;
                    return entry.second;
                }

            cache->emplace_back(d, Detail::probe(d));
            last = &cache->back();
            return last->second;
        }

        /** @brief The cached Profile for the device behind @p q. */
        inline const Profile& profile_for(const sycl::queue& q)
        {
            return profile_for(q.get_device());
        }

        /**
         * @brief The work-group size to launch with, in work-items.
         *
         * Clamped to what the device accepts and rounded down to a whole number
         * of sub-groups, because a partial trailing sub-group is dead lanes on
         * every device that has sub-groups at all.
         */
        [[nodiscard]] inline std::size_t work_group_size(const Profile& p) noexcept
        {
            const std::size_t target = p.is_cpu ? K_CPU_WORK_GROUP_TARGET
                                                : K_GPU_WORK_GROUP_TARGET;
            const std::size_t clamped = std::min(target, p.max_work_group_size);
            const std::size_t whole = (clamped / p.sub_group_size) * p.sub_group_size;
            return whole ? whole : std::size_t{1};
        }

        /**
         * @struct LocalTileDemand
         * @brief Holds what a kernel that stages a tile in work-group local memory needs from a device.
         */
        struct LocalTileDemand
        {
            /** @brief Bytes of local memory one work-group allocates. */
            std::size_t local_bytes = 0;

            /** @brief Work-items in one work-group. */
            std::size_t work_group_items = 1;
        };

        /**
         * @brief Returns whether a device can host a work-group with demand's local memory and size.
         */
        [[nodiscard]] inline bool hosts_local_tile(const Profile& p,
                                                   const LocalTileDemand& demand) noexcept
        {
            return p.local_mem_bytes >= demand.local_bytes &&
                   p.max_work_group_size >= demand.work_group_items;
        }

        /**
         * @brief Holds the launch geometry of a one-dimensional grid-stride kernel.
         * @see modules/foundation/launch/README.md
         */
        struct Grid1D
        {
            /** @brief Total work-items launched; always a multiple of @ref local_size. */
            std::size_t global_size = 1;

            /** @brief Work-items per work-group. */
            std::size_t local_size = 1;

            /** @brief The nd_range to hand to `parallel_for`. */
            [[nodiscard]] sycl::nd_range<1> nd_range() const noexcept
            {
                return sycl::nd_range<1>(sycl::range<1>(global_size),
                                         sycl::range<1>(local_size));
            }

            /** @brief The step a grid-stride loop advances by; equals @ref global_size. */
            [[nodiscard]] std::size_t stride() const noexcept { return global_size; }

            /** @brief Number of work-groups in the launch. */
            [[nodiscard]] std::size_t groups() const noexcept
            {
                return global_size / local_size;
            }
        };

        /**
         * @brief Returns the work-group size a flat, barrier-free 1-D launch uses.
         * @return A size the device accepts; never zero.
         * @see modules/foundation/launch/README.md
         */
        [[nodiscard]] inline std::size_t flat_work_group_size(const Profile& p,
                                                              std::size_t n) noexcept
        {
            if (!p.is_cpu)
                return work_group_size(p);

            const std::size_t per_thread = (n + p.compute_units - 1) / p.compute_units;
            const std::size_t floor_size = std::min(K_CPU_WORK_GROUP_TARGET,
                                                    p.max_work_group_size);
            const std::size_t wanted = std::clamp(per_thread, floor_size,
                                                  p.max_work_group_size);

            // Only whole sub-groups are kept; a partial trailing sub-group is dead lanes; see modules/foundation/launch/README.md.
            const std::size_t whole = (wanted / p.sub_group_size) * p.sub_group_size;
            return whole ? whole : std::size_t{1};
        }

        /**
         * @brief Plans a grid-stride launch over n elements.
         * @param unroll Elements each work-item handles per stride step: 1 for scalar, 4 for 128-bit accesses.
         * @return A geometry whose global size is a non-zero multiple of the local size, even for n == 0.
         * @see modules/foundation/launch/README.md
         */
        [[nodiscard]] inline Grid1D plan_1d(const Profile& p, std::size_t n,
                                            std::size_t unroll = 1) noexcept
        {
            const std::size_t local = flat_work_group_size(p, n);
            const std::size_t per_group = local * (unroll ? unroll : 1);

            const std::size_t to_cover = (n + per_group - 1) / per_group;

            // A CPU grid covers the whole problem; a GPU grid is capped; see modules/foundation/launch/README.md.
            std::size_t groups = to_cover;
            if (!p.is_cpu)
            {
                const std::size_t resident = p.compute_units * K_WAVES_PER_COMPUTE_UNIT;
                groups = std::min(groups, resident);
            }
            if (groups == 0)
                groups = 1;

            return Grid1D{groups * local, local};
        }

        /**
         * @brief Returns how many elements of element_bytes to load in one access.
         * @return A `sycl::vec` width of 1, 2, 4 or 8; always 1 on a CPU.
         * @see modules/foundation/launch/README.md
         */
        [[nodiscard]] inline std::size_t vector_width(const Profile& p,
                                                      std::size_t element_bytes) noexcept
        {
            if (p.is_cpu || element_bytes == 0)
                return 1;
            if (element_bytes >= K_VECTOR_TRANSACTION_BYTES)
                return 1;
            const std::size_t width = K_VECTOR_TRANSACTION_BYTES / element_bytes;
            return width ? width : std::size_t{1};
        }

    } // namespace Device
} // namespace SushiBLAS
