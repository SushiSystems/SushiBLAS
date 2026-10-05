/****************************************************************/
/* device_policy.hpp                                            */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file device_policy.hpp
 * @brief What the device is, expressed as facts a kernel can dispatch on.
 *
 * The fastest CPU kernel and the fastest GPU kernel are not the same kernel,
 * and for the two shapes that matter most here they are close to opposites.
 * A GPU wants a large work-group cooperatively staging tiles through
 * scratchpad memory, bank-padded, with each work-item holding a wide register
 * block. A CPU wants none of that: its "local memory" is ordinary cache
 * reached through the same loads, so staging through it copies the data twice
 * and every `group_barrier` becomes a real synchronisation between OS threads
 * for no benefit. portBLAS ships exactly this split - `gemm_local` for real
 * GPUs, `gemm_no_local` for everything generic - and it is the split this
 * header exists to let a kernel express.
 *
 * The rule this header enforces is that a kernel dispatches on *facts*, never
 * on a vendor string. `has_dedicated_local_memory` is a question about the
 * memory hierarchy; `is_cpu` is a question the SYCL runtime already answers.
 * Neither is "am I on an RTX 4090", which is the kind of test that rots.
 *
 * Nothing here is GEMM-specific or elementwise-specific on purpose. A Profile
 * describes the machine; choosing a tile size or an unroll factor from it is
 * the individual kernel's business, because only the kernel knows how many
 * registers its accumulator costs. Pushing those choices in here would make
 * this file change every time any kernel is tuned.
 *
 * This is a public installed header: SushiAI's fused kernels reach the same
 * policy through the `Dispatch` seam, so a consumer-authored epilogue gets the
 * same launch geometry as a built-in one rather than an accidental
 * `sycl::range` fallback.
 */

#include <deque>
#include <mutex>
#include <vector>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Device
    {
        /**
         * @brief How many resident work-groups per compute unit a launch aims for.
         *
         * A grid-stride kernel does not need one work-item per element; it needs
         * enough work-groups to fill the machine and hide memory latency, and
         * anything beyond that is launch overhead. Eight per compute unit is the
         * usual occupancy target for a bandwidth-bound kernel on a discrete GPU:
         * enough concurrent waves that a stalled one always has a neighbour ready,
         * without the register pressure of chasing maximum occupancy.
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
         * @struct Profile
         * @brief The device properties a kernel is allowed to dispatch on.
         *
         * A plain aggregate of answers, queried once and cached. Everything here
         * comes straight from `sycl::info::device` except
         * @ref has_dedicated_local_memory, which is a judgement the raw query
         * cannot make on its own - see that field.
         */
        struct Profile
        {
            /** @brief True when the SYCL runtime classifies the device as a CPU. */
            bool is_cpu = true;

            /**
             * @brief True when scratchpad memory is physically separate storage.
             *
             * `local_mem_type` alone is not enough to answer this. An OpenCL CPU
             * device reports `local_mem_type::local` and a plausible
             * `local_mem_size`, but backs both with the same cache line the global
             * pointer already reads through. Staging a tile there is a pure extra
             * copy, and the barriers guarding it are pure loss. So a kernel that
             * asks "is it worth tiling?" must ask this, not `local_mem_type`.
             */
            bool has_dedicated_local_memory = false;

            /** @brief True when the device supports double precision natively. */
            bool supports_fp64 = false;

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

            /** @brief Global memory cache line, in bytes; 0 when the device won't say. */
            std::size_t global_mem_cache_line = 0;
        };

        namespace Detail
        {
            /** @brief Queries every Profile field from a device. Not cached; call profile_for. */
            inline Profile probe(const sycl::device& d)
            {
                Profile p;
                p.is_cpu = d.is_cpu();
                p.supports_fp64 = d.has(sycl::aspect::fp64);
                p.local_mem_bytes = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::local_mem_size>());
                p.max_work_group_size = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::max_work_group_size>());
                p.compute_units = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::max_compute_units>());
                p.global_mem_cache_line = static_cast<std::size_t>(
                    d.get_info<sycl::info::device::global_mem_cache_line_size>());

                const bool declares_local =
                    d.get_info<sycl::info::device::local_mem_type>() ==
                    sycl::info::local_mem_type::local;
                p.has_dedicated_local_memory = declares_local && !p.is_cpu &&
                                               p.local_mem_bytes > 0;

                // Widest reported sub-group: 32 on NVIDIA, 64 on AMD wave64, and on
                // Intel the widest of {8,16,32}, which is the one that amortises a
                // shuffle reduction best. An empty list would be a spec violation,
                // but a zero here would divide by zero downstream, so it is guarded.
                const std::vector<std::size_t> widths =
                    d.get_info<sycl::info::device::sub_group_sizes>();
                p.sub_group_size = widths.empty()
                    ? std::size_t{1}
                    : *std::max_element(widths.begin(), widths.end());
                if (p.sub_group_size == 0) p.sub_group_size = 1;

                if (p.max_work_group_size == 0) p.max_work_group_size = 1;
                if (p.compute_units == 0) p.compute_units = 1;

                return p;
            }
        } // namespace Detail

        /**
         * @brief The cached Profile for @p d, probing the device on first sight.
         *
         * Called on every task execution, so it must not re-query: a replayed
         * graph runs the same fifty nodes thousands of times and eight
         * `get_info` round trips per node per step is real overhead against a
         * few-microsecond kernel.
         *
         * @param d The device to describe.
         * @return A reference valid for the rest of the process.
         */
        inline const Profile& profile_for(const sycl::device& d)
        {
            // A std::deque, not a vector: a reference handed to an earlier caller
            // must stay valid when a second device is probed, and only deque
            // guarantees that across push_back.
            //
            // Deliberately leaked. The entries hold sycl::device handles, and a
            // static destructor running at exit is not ordered against the SYCL
            // runtime tearing itself down - releasing a device handle afterwards
            // is a crash on some backends. One small permanent allocation is the
            // standard price for not having that race.
            static std::deque<std::pair<sycl::device, Profile>>* cache =
                new std::deque<std::pair<sycl::device, Profile>>();
            static std::mutex guard;

            const std::lock_guard<std::mutex> lock(guard);
            for (auto& entry : *cache)
                if (entry.first == d) return entry.second;

            cache->emplace_back(d, Detail::probe(d));
            return cache->back().second;
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
         * @struct Grid1D
         * @brief A launch geometry for a one-dimensional grid-stride kernel.
         *
         * The kernel this describes walks its elements with a stride loop rather
         * than assuming one work-item per element:
         *
         * @code
         * const auto grid = Device::plan_1d(profile, n, unroll);
         * h.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it) {
         *     for (std::size_t i = it.get_global_id(0); i < n; i += grid_stride)
         *         out[i] = op(in[i]);
         * });
         * @endcode
         *
         * The `i < n` test is therefore the bounds check a padded grid would
         * otherwise need, so there is no divergent tail work-group whatever
         * @ref global_size turns out to be.
         *
         * On a GPU @ref global_size is deliberately *not* `n` rounded up: it is
         * capped at roughly what the machine can hold resident and the loop
         * covers the rest, which stops a large tensor from scheduling a grid far
         * bigger than the device can run at once. On a CPU it is `n` rounded up,
         * so the loop body runs once - see @ref plan_1d for the measurement that
         * settled that split.
         *
         * The traversal order is fixed by the geometry, so two runs of the same
         * kernel on the same device touch elements in the same order - which is
         * what makes a deterministic reduction built on this reproducible.
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
         * @brief The work-group size a flat, barrier-free 1-D launch should use.
         *
         * Not @ref work_group_size, and the difference is the point.
         * @ref work_group_size answers "how wide a group can cooperate cheaply?",
         * which is a question about barriers, and its CPU answer is deliberately
         * small because a barrier there is a rendezvous between OS threads. A
         * flat elementwise launch has no barriers at all, so nothing about that
         * answer applies to it; what it pays for instead is *work-group count*,
         * because on an OpenCL CPU backend each group is a scheduling unit with
         * real per-group cost. Aiming at one group per hardware thread and
         * letting the group get as wide as it needs to be turns that cost from
         * "once per 64 elements" into "once per thread".
         *
         * The GPU answer is just @ref work_group_size: there, group count is how
         * occupancy is expressed, and 256 work-items is already the right width.
         *
         * @param p The target device's profile.
         * @param n How many elements the launch must cover.
         * @return A work-group size the device will accept; never zero.
         */
        [[nodiscard]] inline std::size_t flat_work_group_size(const Profile& p,
                                                              std::size_t n) noexcept
        {
            if (!p.is_cpu) return work_group_size(p);

            const std::size_t per_thread = (n + p.compute_units - 1) / p.compute_units;
            const std::size_t floor_size = std::min(K_CPU_WORK_GROUP_TARGET,
                                                    p.max_work_group_size);
            const std::size_t wanted = std::clamp(per_thread, floor_size,
                                                  p.max_work_group_size);

            // Whole sub-groups only, for the same reason work_group_size rounds:
            // a partial trailing sub-group is dead lanes.
            const std::size_t whole = (wanted / p.sub_group_size) * p.sub_group_size;
            return whole ? whole : std::size_t{1};
        }

        /**
         * @brief Plans a grid-stride launch over @p n elements.
         *
         * @param p The target device's profile.
         * @param n How many elements the kernel must cover.
         * @param unroll Elements each work-item handles per stride step; 1 for a
         *        plain scalar loop, 4 when the kernel issues 128-bit vector
         *        accesses. Only affects how large the grid needs to be.
         * @return A geometry whose global size is a multiple of the local size
         *         and never zero, even for @p n == 0.
         */
        [[nodiscard]] inline Grid1D plan_1d(const Profile& p, std::size_t n,
                                            std::size_t unroll = 1) noexcept
        {
            const std::size_t local = flat_work_group_size(p, n);
            const std::size_t per_group = local * (unroll ? unroll : 1);

            const std::size_t to_cover = (n + per_group - 1) / per_group;

            // A CPU gets a grid that covers the whole problem, so the stride loop
            // body runs at most once and the launch is "one work-item per element"
            // with an explicit work-group size. That is the opposite of the GPU
            // choice below, and it is a locality argument rather than an occupancy
            // one: a CPU backend hands each OS thread a contiguous run of
            // work-groups, which its prefetchers stream through, and capping the
            // grid replaces that with a strided walk - each thread touching one
            // group's worth of elements, then jumping past every other thread's
            // share to reach its next one.
            //
            // Measured on this project's OpenCL CPU device (12 compute units,
            // 8192 max work-group), 200 launches of a 1M-element elementwise add:
            // 28.3 ms capped at one group per compute unit, 9.7 ms uncapped with
            // the work-group sizing above, against 8.2 ms for the bare
            // `sycl::range<1>` this replaced. The residual gap is nd_range setup
            // the CPU backend charges and the flat launch does not; at the small
            // and medium sizes this library actually launches at (1K-64K), the
            // planned shape is the faster of the two by 5-7%.
            //
            // A GPU keeps the cap. Consecutive work-items still read consecutive
            // elements at every step of the stride loop, so each step is as
            // coalesced as a flat launch would be, oversubscribing resident
            // work-groups is how memory latency gets hidden, and the cap is what
            // stops a large tensor from scheduling a grid far larger than the
            // device can hold resident at once.
            std::size_t groups = to_cover;
            if (!p.is_cpu)
            {
                const std::size_t resident = p.compute_units * K_WAVES_PER_COMPUTE_UNIT;
                groups = std::min(groups, resident);
            }
            if (groups == 0) groups = 1;

            return Grid1D{groups * local, local};
        }

        /**
         * @brief How many elements of @p element_bytes to load in one access.
         *
         * Returns 1 on a CPU. Not an oversight: the OpenCL CPU backend already
         * vectorises *across* work-items, packing neighbouring items into SIMD
         * lanes, and a hand-written `sycl::vec` fights that by consuming the same
         * lanes for a single item. On a GPU there is no such implicit
         * vectorisation, and a 128-bit access is one transaction where four
         * 32-bit accesses are four.
         *
         * @param p The target device's profile.
         * @param element_bytes `sizeof` the element type.
         * @return A `sycl::vec` width - 1, 2, 4 or 8 - never 0.
         */
        [[nodiscard]] inline std::size_t vector_width(const Profile& p,
                                                      std::size_t element_bytes) noexcept
        {
            if (p.is_cpu || element_bytes == 0) return 1;
            if (element_bytes >= K_VECTOR_TRANSACTION_BYTES) return 1;
            const std::size_t width = K_VECTOR_TRANSACTION_BYTES / element_bytes;
            return width ? width : std::size_t{1};
        }

    } // namespace Device
} // namespace SushiBLAS
