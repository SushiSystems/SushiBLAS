/**************************************************************************/
/* orthogonal.cpp                                                         */
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
 * @file orthogonal.cpp
 * @brief Defines RandomOps::orthogonal, which records the orthogonal initialisation of a matrix.
 * @author Mustafa Garip
 */

#include <vector>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
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

    sycl::event RandomOps::orthogonal(Tensor& t, double gain)
    {
        const char* name = "random.orthogonal";

        SB_THROW_IF(t.rank != 2, "Operation '{}' requires a rank-2 tensor, got rank {}.", name,
                    t.rank);
        Dispatch::require_random_dtype(t, Dispatch::Domain::REAL, Dispatch::Sampling::DRAWS_VALUES,
                                       name);

        const int64_t rows = t.shape[0];
        const int64_t cols = t.shape[1];
        SB_THROW_IF(rows <= 0 || cols <= 0,
                    "Operation '{}' requires a non-empty matrix, got {}x{}.", name, rows, cols);
        Dispatch::require_storage(t, name, "t");
        Dispatch::require_contiguous(t, name, "t");

        // The matrix is filled with N(0,1) before the sweep; see src/README.md.
        normal(t, 0.0, 1.0);

        // The shorter axis is orthonormalized, read through t.strides; see src/README.md.
        const bool by_rows = (rows <= cols);
        Kernels::SYCL::Random::VectorSet set;
        set.count = by_rows ? rows : cols;
        set.length = by_rows ? cols : rows;
        set.elem_stride = by_rows ? t.strides[1] : t.strides[0];
        set.vec_stride = by_rows ? t.strides[0] : t.strides[1];

        void* ptr = t.data();
        std::vector<SushiRuntime::Core::ResourceRegion> rw = {t.region()};

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = name;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.orthogonal"_op;
        meta.set_param(0, gain);

        engine_.get_graph().add_task(meta, rw, rw,
            [dtype = t.dtype, ptr, set, gain, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                SB_LOG_INFO("RandomOps: {} ({} vectors of length {}, gain: {:.4f})", name, set.count, set.length, gain);

                sycl::event done;
                Dispatch::with_element_type<Dispatch::Domain::REAL>(dtype, "RNG operation", name, [&](auto tag)
                {
                    using T = typename decltype(tag)::type;
                    if constexpr (std::is_same_v<T, sycl::half>)
                        SB_THROW_IF(true, "RNG operation '{}' does not support dtype HALF.", name);
                    else
                        done = Kernels::SYCL::Random::orthonormalize(q, deps, static_cast<T*>(ptr), set,
                                                                     static_cast<T>(gain));
                });
                return done;
            }
        );

        return sycl::event();
    }
} // namespace SushiBLAS
