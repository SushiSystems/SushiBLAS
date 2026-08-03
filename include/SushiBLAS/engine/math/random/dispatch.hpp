/**************************************************************************/
/* dispatch.hpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <type_traits>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Task-metadata slot holding the RNG seed, stored as raw `uint64_t` bits.
         *
         * The seed and offset sit in the two highest slots rather than immediately after the
         * distribution's own parameters so that their position is the same for every random
         * op, whatever its arity - a profiler or replay tool reading the metadata does not
         * have to know which distribution produced the task in order to find them.
         *
         * They used to be written to slots 10 and 11. `TaskMetadata` holds eight, and
         * `set_param` ignores an out-of-range index without a word, so every random task
         * recorded so far carried a zero seed and a zero offset in its metadata. The kernels
         * themselves were unaffected (they capture seed/offset by value, not through the
         * metadata), so the bug never changed a number - it only blanked the record of which
         * draw a task made.
         */
        inline constexpr int K_RNG_SEED_PARAM = 6;

        /** @brief Task-metadata slot holding the per-task RNG offset, as raw `uint64_t` bits. */
        inline constexpr int K_RNG_OFFSET_PARAM = 7;

        /**
         * @brief Distribution parameters a random op may record, i.e. the slots left over
         * once @ref K_RNG_SEED_PARAM and @ref K_RNG_OFFSET_PARAM are reserved.
         */
        inline constexpr size_t K_MAX_RNG_PARAMS = 6;

        /**
         * @brief Stamps the seed/offset pair a random task drew from into its metadata.
         *
         * @param meta The metadata block to annotate.
         * @param seed The engine's current RNG seed.
         * @param offset The per-task counter offset the engine handed out.
         */
        inline void set_rng_params(SushiRuntime::Graph::TaskMetadata& meta, uint64_t seed, uint64_t offset)
        {
            static_assert(K_RNG_OFFSET_PARAM < static_cast<int>(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS),
                          "RNG metadata slots must fit inside TaskMetadata's inline parameter array.");
            meta.set_param(K_RNG_SEED_PARAM, seed);
            meta.set_param(K_RNG_OFFSET_PARAM, offset);
        }

        /**
         * @brief Submits one counter-based fill: `out[i] = draw(i)` for every i in [0, @p n).
         *
         * Every distribution in this library has that shape - the value at an index is a
         * pure function of the index, because a counter-based generator is what makes a
         * fill reproducible from `(seed, offset, index)` regardless of how the work was
         * scheduled. Each of the twelve `RandomOps` entry points used to spell the
         * submission out for itself: `q.submit`, `depends_on(deps)`, and a bare
         * `parallel_for(sycl::range<1>(size), ...)`. That last part is the one that
         * mattered - it was the runtime's default geometry, not this library's, and it
         * was the same twelve times over.
         *
         * The launch now comes from @ref SushiBLAS::Launch::indexed, so a distribution
         * gets the same device-dependent shape as an elementwise op and gets it without
         * knowing that shape exists. @ref SushiBLAS::Launch::map is not the right
         * primitive here despite the contiguous write, because there is no input array to
         * read: the "input" is the index itself.
         *
         * @tparam T The element type being written, deduced.
         * @tparam Draw The generator functor type, deduced.
         * @param q The queue to submit to.
         * @param deps Events this submission must depend on.
         * @param n How many elements to fill.
         * @param out The destination array.
         * @param draw Callable `(uint64_t index) -> T`, invoked once per element on device.
         * @return The submitted work's event.
         *
         * @warning @p draw is inlined into a SYCL device kernel: it must be
         *          device-copyable, capture by value only, hold no host pointers,
         *          allocate nothing and throw nothing.
         */
        template<typename T, typename Draw>
        [[nodiscard]] inline sycl::event submit_random_fill(sycl::queue& q, const std::vector<sycl::event>& deps,
                                                            int64_t n, T* out, Draw draw)
        {
            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                Launch::indexed(h, q, static_cast<std::size_t>(n), [=](std::size_t i)
                {
                    out[i] = draw(static_cast<uint64_t>(i));
                });
            });
        }

        /**
         * @brief Records a random-fill op, resolving the tensor's dtype to a typed pointer.
         *
         * The generic seam behind every `RandomOps` distribution. It owns the parts every
         * distribution shares - graph bookkeeping, drawing a fresh counter offset from the
         * engine so two tasks never reuse a Philox (key, counter) pair, and the dtype switch -
         * and leaves the actual draw to the caller's functor.
         *
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor to fill.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param params The distribution's own parameters, recorded in metadata slots 0..n-1.
         * @param task_func Callable `(auto scalar_type, sycl::queue&, uint64_t seed,
         *        uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps)
         *        -> sycl::event`, generic over the runtime dtype via `decltype(scalar_type)`.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning This is a low-level generic seam. Whatever `task_func` submits runs on the
         *          device: any nested lambda must be device-copyable, capture by value, hold
         *          no host pointers, allocate nothing and throw nothing. Draw through
         *          `Kernels::SYCL::Random` rather than a stateful host RNG - the counter-based
         *          generator is what makes a fill reproducible from (seed, offset, index)
         *          alone, independent of how the work is scheduled.
         * @throws std::runtime_error for HALF (too little mantissa for the distributions'
         *         inverse-CDF and Box-Muller transforms), for an unrecognized dtype, or if
         *         @p params exceeds @ref K_MAX_RNG_PARAMS.
         */
        template<typename Func>
        sycl::event execute_random(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, const std::vector<double>& params, Func&& task_func)
        {
            // At record time, not inside the task. The switch below rejects an unsupported
            // dtype too, but only once the graph is executing, by which point the caller is
            // several frames away from the line that queued the op and the graph already
            // holds a task that can never run.
            SB_THROW_IF(Core::is_integer_dtype(t.dtype) || t.dtype == Core::DataType::HALF,
                        "RNG operation '{}' does not support dtype {}. A continuous "
                        "distribution needs a fractional part to land in, which the integer "
                        "types do not have, and more mantissa than HALF supplies for the "
                        "inverse-CDF and Box-Muller transforms. Use RandomOps::constant for a "
                        "deterministic fill of either.",
                        name, Core::to_string(t.dtype));

            const int64_t size = t.num_elements;
            void* ptr = t.data_or_null();

            std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
            if (ptr) writes.push_back(t.region());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name, SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS - K_MAX_RNG_PARAMS);

            const uint64_t seed = engine.get_seed();
            const uint64_t offset = engine.get_and_increment_rng_offset();
            set_rng_params(meta, seed, offset);

            engine.get_graph().add_task(meta, reads, writes,
                [dtype=t.dtype, size, seed, offset, task_func, pT = ptr, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("RandomOps: {} ({} elements, seed: {}, offset: {})", name, size, seed, offset);

                    switch (dtype)
                    {
                        case Core::DataType::FLOAT32:
                            return task_func(float{}, q, seed, offset, size, static_cast<float*>(pT), deps);
                        case Core::DataType::FLOAT64:
                            return task_func(double{}, q, seed, offset, size, static_cast<double*>(pT), deps);
                        case Core::DataType::COMPLEX32:
                            return task_func(std::complex<float>{}, q, seed, offset, size, static_cast<std::complex<float>*>(pT), deps);
                        case Core::DataType::COMPLEX64:
                            return task_func(std::complex<double>{}, q, seed, offset, size, static_cast<std::complex<double>*>(pT), deps);
                        // HALF has too little mantissa for the inverse-CDF and Box-Muller
                        // transforms; the integer dtypes have no fractional part for a
                        // continuous distribution to land in at all. `RandomOps::constant`
                        // serves both of those and does not come through here.
                        case Core::DataType::HALF:
                        case Core::DataType::INT32:
                        case Core::DataType::INT64:
                            SB_THROW_IF(true, "RNG operation '{}' does not support dtype {}.", name, Core::to_string(dtype));
                            return sycl::event();
                        default:
                            SB_THROW_IF(true, "RNG operation '{}' received an unrecognized dtype ({}).",
                                        name, static_cast<int>(dtype));
                            return sycl::event();
                    }
                }
            );
            return sycl::event();
        }

    } // namespace Dispatch
} // namespace SushiBLAS
