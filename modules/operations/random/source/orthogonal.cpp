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
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/random/dispatch.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/support/orthonormal_sweep.hpp>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle RandomOps::orthogonal(Tensor& t, double gain)
    {
        const char* name = "random.orthogonal";

        Dispatch::require_rank(t, 2, name, "t");
        Dispatch::require_random_dtype(t, Dispatch::Domain::Real, Dispatch::Sampling::DrawsValues,
                                       name);

        const int64_t rows = t.shape[0];
        const int64_t cols = t.shape[1];
        Dispatch::require_min_elements(t, 1, name, "t");
        Dispatch::require_storage(t, name, "t");
        Dispatch::require_device_dtype(t, name, "t");
        Dispatch::require_contiguous(t, name, "t");

        OperandSet operands;
        operands.updates(t);
        recorder_.require_host_reachable(operands, "RNG operation", name);

        // The matrix is filled with N(0,1) before the sweep; see modules/operations/README.md.
        normal(t, 0.0, 1.0);

        // The shorter axis is orthonormalized, read through t.strides; see modules/operations/README.md.
        const bool by_rows = (rows <= cols);
        Kernels::VectorSet set;
        set.count = by_rows ? rows : cols;
        set.length = by_rows ? cols : rows;
        set.elem_stride = by_rows ? t.strides[1] : t.strides[0];
        set.vec_stride = by_rows ? t.strides[0] : t.strides[1];

        void* ptr = t.data();

        TaskDescription meta;
        meta.name = name;
        meta.op_id = "random.orthogonal"_op;
        meta.set_param(0, gain);

        return recorder_.record_pair(meta, operands, "RNG operation",
            [dtype = t.dtype, ptr, set, gain, name]
            (const Internal::KernelPairDispatcher& pair, sycl::queue& q,
             const std::vector<sycl::event>& deps) -> sycl::event
            {
                SB_LOG_INFO("RandomOps: {} ({} vectors of length {}, gain: {:.4f})", name, set.count, set.length, gain);

                sycl::event done;
                Dispatch::with_element_type<Dispatch::Domain::Real>(dtype, "RNG operation", name, [&](auto tag)
                {
                    using T = typename decltype(tag)::type;
                    if constexpr (std::is_same_v<T, sycl::half>)
                    {
                        SB_THROW_IF(true, "RNG operation '{}' does not support dtype HALF.", name);
                    }
                    else
                    {
                        T* vectors = static_cast<T*>(ptr);
                        const T scale = static_cast<T>(gain);

                        done = pair.run(q, deps,
                            [vectors, set, scale]
                            {
                                Kernels::Reference::Random::orthonormalize(vectors, set, scale);
                            },
                            [vectors, set, scale](sycl::queue& q2,
                                                  const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                return Kernels::SYCL::Random::orthonormalize(q2, deps2, vectors,
                                                                             set, scale);
                            });
                    }
                });
                return done;
            });
    }
} // namespace SushiBLAS
