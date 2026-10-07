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
#include <string>
#include <tuple>
#include <vector>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
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
         * @brief Holds how many distribution parameters a random op may record: the slots left
         * once @ref K_RNG_SEED_PARAM and @ref K_RNG_OFFSET_PARAM are reserved.
         */
        inline constexpr size_t K_MAX_RNG_PARAMS = 6;

        /// @brief Stamps the seed and offset a random task draws from into its metadata.
        inline void set_rng_params(SushiRuntime::Graph::TaskMetadata& meta, uint64_t seed,
                                   uint64_t offset)
        {
            static_assert(K_RNG_OFFSET_PARAM <
                              static_cast<int>(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS),
                          "RNG metadata slots must fit inside TaskMetadata's parameter array.");
            meta.set_param(K_RNG_SEED_PARAM, seed);
            meta.set_param(K_RNG_OFFSET_PARAM, offset);
        }

        /**
         * @brief Names what a random op does with element values, which decides whether it accepts HALF.
         * @see include/SushiBLAS/engine/math/README.md
         */
        enum class Sampling : uint8_t
        {
            DRAWS_VALUES, /**< Computes new values; HALF has too little mantissa and is refused. */
            MOVES_VALUES  /**< Fills or permutes values; every dtype of its domain is accepted. */
        };

        /**
         * @brief Reports whether a random op of @p domain and @p sampling accepts @p dtype.
         * @see include/SushiBLAS/engine/math/README.md
         */
        [[nodiscard]] inline constexpr bool random_admits(Domain domain, Sampling sampling,
                                                          Core::DataType dtype) noexcept
        {
            if (sampling == Sampling::DRAWS_VALUES && dtype == Core::DataType::HALF)
                return false;
            return domain_admits(domain, domain_of(dtype));
        }

        /**
         * @brief Throws std::invalid_argument naming the op and t's dtype unless random_admits holds.
         * @see include/SushiBLAS/engine/math/README.md
         */
        inline void require_random_dtype(const Tensor& t, Domain domain, Sampling sampling, const char* name)
        {
            if (random_admits(domain, sampling, t.dtype))
                return;

            constexpr Core::DataType k_dtypes[] = {
                Core::DataType::HALF, Core::DataType::FLOAT32, Core::DataType::FLOAT64,
                Core::DataType::COMPLEX32, Core::DataType::COMPLEX64,
                Core::DataType::INT32, Core::DataType::INT64};

            std::string accepted;
            for (Core::DataType dtype : k_dtypes)
            {
                if (!random_admits(domain, sampling, dtype))
                    continue;
                if (!accepted.empty())
                    accepted += ", ";
                accepted += Core::to_string(dtype);
            }

            Detail::throw_operand_error(name, "t", "has dtype {}; {} accepts {} only",
                                        Core::to_string(t.dtype), name, accepted);
        }

        /**
         * @brief Submits one counter-based fill in blocks: `out[b*K + k] = draw_block(b)[k]` below n.
         * @param draw_block Callable `(uint64_t block) -> std::array<T, K>`, inlined into a SYCL device kernel.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T, typename DrawBlock>
        [[nodiscard]] inline sycl::event submit_random_fill(sycl::queue& q, const std::vector<sycl::event>& deps,
                                                            int64_t n, T* out, DrawBlock draw_block)
        {
            constexpr std::size_t k_per_block = std::tuple_size<decltype(draw_block(uint64_t{0}))>::value;
            const std::size_t count = static_cast<std::size_t>(n);
            const std::size_t blocks = (count + k_per_block - 1) / k_per_block;

            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                Launch::indexed(h, q, blocks, [=](std::size_t b)
                {
                    const auto values = draw_block(static_cast<uint64_t>(b));
                    for (std::size_t k = 0; k < k_per_block; ++k)
                    {
                        const std::size_t i = b * k_per_block + k;
                        if (i < count)
                            out[i] = static_cast<T>(values[k]);
                    }
                });
            });
        }

        namespace Detail
        {
            /**
             * @brief Runs one random-fill task body for the element type of a with_element_type arm.
             * @tparam Func The task body: `(T, queue&, seed, offset, size, T*, deps) -> sycl::event`.
             */
            template <typename Func>
            struct RandomArm
            {
                /** @brief Holds the task body the arm calls. */
                const Func& task_func;
                /** @brief Holds the queue the fill is submitted to. */
                sycl::queue& q;
                /** @brief Holds the stream seed recorded with the task. */
                uint64_t seed;
                /** @brief Holds the stream offset recorded with the task. */
                uint64_t offset;
                /** @brief Holds the element count of the tensor. */
                int64_t size;
                /** @brief Holds the tensor's base address. */
                void* ptr;
                /** @brief Holds the events the fill depends on. */
                const std::vector<sycl::event>& deps;
                /** @brief Holds the operation name for messages. */
                const char* name;
                /** @brief Receives the event the task body returns. */
                sycl::event& done;

                /**
                 * @brief Submits the fill for the element type named by the tag.
                 * @throws std::runtime_error for HALF.
                 */
                template <typename Tag>
                void operator()(Tag) const
                {
                    using T = typename Tag::type;
                    if constexpr (std::is_same_v<T, sycl::half>)
                        SB_THROW_IF(true, "RNG operation '{}' does not support dtype HALF.", name);
                    else
                        done = task_func(T{}, q, seed, offset, size, static_cast<T*>(ptr), deps);
                }
            };
        } // namespace Detail

        /**
         * @brief Records a random-fill op and resolves the tensor's dtype to a typed pointer.
         * @tparam Dom The element families the distribution draws; HALF is always refused.
         * @param params The distribution's parameters, recorded in metadata slots 0..n-1.
         * @param task_func Submits the fill for one element type of Dom; whatever it submits runs on the device.
         * @throws std::invalid_argument for a dtype outside Dom, HALF, no storage, a non-contiguous t, or under FORCE_REFERENCE.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<Domain Dom, typename Func>
        TaskHandle execute_random(TaskRecorder& recorder, RngStream& stream, Tensor& t,
                                  const char* name, SushiRuntime::Graph::OpID op_id,
                                  const std::vector<double>& params, Func&& task_func)
        {
            require_random_dtype(t, Dom, Sampling::DRAWS_VALUES, name);
            require_storage(t, name, "t");
            require_contiguous(t, name, "t");
            recorder.require_sycl_path("RNG operation", name);

            const int64_t size = t.num_elements;
            void* ptr = t.data_or_null();

            OperandSet operands;
            operands.writes(t);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name, SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS - K_MAX_RNG_PARAMS);

            const RngPosition position = stream.reserve(1);
            const uint64_t seed = position.seed;
            const uint64_t offset = position.offset;
            set_rng_params(meta, seed, offset);

            return recorder.record(meta, operands, "RNG operation",
                [dtype=t.dtype, size, seed, offset, task_func, pT = ptr, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("RandomOps: {} ({} elements, seed: {}, offset: {})", name, size, seed, offset);

                    sycl::event done;
                    const Detail::RandomArm<std::remove_reference_t<Func>> arm{
                        task_func, q, seed, offset, size, pT, deps, name, done};
                    with_element_type<Dom>(dtype, "RNG operation", name, arm);
                    return done;
                });
        }

    } // namespace Dispatch
} // namespace SushiBLAS
