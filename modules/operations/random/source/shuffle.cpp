/**************************************************************************/
/* shuffle.cpp                                                            */
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
 * @file shuffle.cpp
 * @brief Defines RandomOps::shuffle, which records the in-place shuffle of a tensor's elements.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/random/dispatch.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <SushiBLAS/random_kernels/random.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Permutes size elements in place by Fisher-Yates over one (seed, offset) stream.
         * @see modules/operations/README.md
         */
        template<typename T>
        struct FisherYates
        {
            /** @brief Holds the first element. */
            T* p;
            /** @brief Holds the element count. */
            int64_t size;
            /** @brief Holds the stream seed recorded with the task. */
            uint64_t seed;
            /** @brief Holds the stream offset recorded with the task. */
            uint64_t offset;

            /** @brief Runs the permutation; the host path and the device task both call it. */
            void operator()() const
            {
                for (int64_t i = size - 1; i > 0; --i)
                {
                    const int64_t j = static_cast<int64_t>(Kernels::SYCL::Random::draw_index(
                        seed, offset, static_cast<uint64_t>(i), static_cast<uint64_t>(i) + 1));

                    const T tmp = p[i];
                    p[i] = p[j];
                    p[j] = tmp;
                }
            }
        };

        /**
         * @brief Runs the Fisher-Yates permutation of size elements of T on the host or as one device task.
         * @return The submitted event, or an empty one under ForceReference.
         * @see modules/operations/README.md
         */
        template<typename T>
        sycl::event shuffle_dispatch(const Internal::KernelPairDispatcher& pair, sycl::queue& q,
                                     T* p, int64_t size, uint64_t seed, uint64_t offset,
                                     const std::vector<sycl::event>& deps)
        {
            const FisherYates<T> permute{p, size, seed, offset};

            return pair.run(q, deps, permute,
                [permute](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                {
                    return q2.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps2);
                        h.single_task(permute);
                    });
                });
        }
    }

    TaskHandle RandomOps::shuffle(Tensor& t)
    {
        const char* name = "random.shuffle";

        Dispatch::require_random_dtype(t, Dispatch::Domain::RealAndComplex, Dispatch::Sampling::MovesValues, name);
        Dispatch::require_storage(t, name, "t");
        Dispatch::require_device_dtype(t, name, "t");
        Dispatch::require_contiguous(t, name, "t");

        const int64_t size = t.num_elements;
        void* ptr = t.data_or_null();

        // A shuffle reads what it writes, so the region is read-write; see modules/operations/README.md.
        OperandSet operands;
        operands.updates(t);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = name;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.shuffle"_op;

        recorder_.require_host_reachable(operands, "RNG operation", name);
        const RngPosition position = rng_.reserve(1);
        const uint64_t seed = position.seed;
        const uint64_t offset = position.offset;
        Dispatch::set_rng_params(meta, seed, offset);

        return recorder_.record_pair(meta, operands, "RNG operation",
            [dtype = t.dtype, size, seed, offset, ptr, name]
            (const Internal::KernelPairDispatcher& pair, sycl::queue& q,
             const std::vector<sycl::event>& deps) -> sycl::event
            {
                SB_LOG_INFO("RandomOps: {} ({} elements, seed: {}, offset: {})", name, size, seed, offset);

                sycl::event done;
                Dispatch::with_element_type<Dispatch::Domain::RealAndComplex>(dtype, "RNG operation", name, [&](auto tag)
                {
                    using T = typename decltype(tag)::type;
                    done = shuffle_dispatch(pair, q, static_cast<T*>(ptr), size, seed, offset, deps);
                });
                return done;
            });
    }
} // namespace SushiBLAS
