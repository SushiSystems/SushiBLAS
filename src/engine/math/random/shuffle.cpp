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
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        // Fisher-Yates runs in one sequential task, one instantiation per dtype; see src/README.md.
        template<typename T>
        sycl::event shuffle_dispatch(sycl::queue& q, T* p, int64_t size, uint64_t seed, uint64_t offset,
                                      const std::vector<sycl::event>& deps)
        {
            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.single_task([=]()
                {
                    for (int64_t i = size - 1; i > 0; --i)
                    {
                        int64_t j = static_cast<int64_t>(Kernels::SYCL::Random::draw_index(
                            seed, offset, static_cast<uint64_t>(i), static_cast<uint64_t>(i) + 1));

                        T tmp = p[i];
                        p[i] = p[j];
                        p[j] = tmp;
                    }
                });
            });
        }
    }

    TaskHandle RandomOps::shuffle(Tensor& t)
    {
        const char* name = "random.shuffle";

        Dispatch::require_random_dtype(t, Dispatch::Domain::REAL_AND_COMPLEX, Dispatch::Sampling::MOVES_VALUES, name);
        Dispatch::require_storage(t, name, "t");
        Dispatch::require_contiguous(t, name, "t");

        const int64_t size = t.num_elements;
        void* ptr = t.data_or_null();

        // A shuffle reads what it writes, so the region is read-write; see src/README.md.
        OperandSet operands;
        operands.updates(t);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = name;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.shuffle"_op;

        recorder_.require_sycl_path("RNG operation", name);
        const RngPosition position = rng_.reserve(1);
        const uint64_t seed = position.seed;
        const uint64_t offset = position.offset;
        Dispatch::set_rng_params(meta, seed, offset);

        return recorder_.record(meta, operands, "RNG operation",
            [dtype = t.dtype, size, seed, offset, ptr, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                SB_LOG_INFO("RandomOps: {} ({} elements, seed: {}, offset: {})", name, size, seed, offset);

                sycl::event done;
                Dispatch::with_element_type<Dispatch::Domain::REAL_AND_COMPLEX>(dtype, "RNG operation", name, [&](auto tag)
                {
                    using T = typename decltype(tag)::type;
                    done = shuffle_dispatch(q, static_cast<T*>(ptr), size, seed, offset, deps);
                });
                return done;
            });
    }
} // namespace SushiBLAS
