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
 * @file dispatch.hpp
 * @brief Declares the random-fill executor of the dispatch seam.
 * @author Mustafa Garip
 */

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
         * @brief Holds the task-metadata slot of the RNG seed, stored as raw `uint64_t` bits.
         * @see include/SushiBLAS/engine/math/README.md
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
         * @brief Submits one counter-based fill: `out[i] = draw(i)` for every i in [0, n).
         * @param draw Callable `(uint64_t index) -> T`, inlined into a SYCL device kernel.
         * @see include/SushiBLAS/engine/math/README.md
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
         * @brief Records a random-fill op and resolves the tensor's dtype to a typed pointer.
         * @param params The distribution's parameters, recorded in metadata slots 0..n-1.
         * @param task_func Submits the fill for one element type; whatever it submits runs on the device.
         * @throws std::runtime_error for HALF, an integer or unrecognized dtype, or too many params.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Func>
        sycl::event execute_random(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, const std::vector<double>& params, Func&& task_func)
        {
            // HALF and the integer dtypes are rejected at record time; see include/SushiBLAS/engine/math/README.md.
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
                        // HALF and integers have no continuous draw here; see include/SushiBLAS/engine/math/README.md.
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
