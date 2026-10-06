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

    namespace
    {
        template<typename T>
        sycl::event orthogonal_dispatch(sycl::queue& q, T* p, int64_t count, int64_t length,
                                         int64_t elem_stride, int64_t vec_stride, T gain,
                                         const std::vector<sycl::event>& deps)
        {
            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.single_task([=]()
                {
                    Kernels::SYCL::Random::orthonormalize_mgs(p, count, length, elem_stride, vec_stride, gain);
                });
            });
        }
    }

    sycl::event RandomOps::orthogonal(Tensor& t, double gain)
    {
        const char* name = "random.orthogonal";

        SB_THROW_IF(t.rank != 2, "Operation '{}' requires a rank-2 tensor, got rank {}.", name, t.rank);
        SB_THROW_IF(!t.storage, "Operation '{}' requires a tensor with allocated storage.", name);
        SB_THROW_IF(Core::is_complex_dtype(t.dtype) || t.dtype == Core::DataType::HALF,
                    "Operation '{}' does not support dtype {}: the Gram-Schmidt sweep needs a real inner "
                    "product, and HALF loses orthogonality to rounding before the sweep finishes.",
                    name, Core::to_string(t.dtype));

        const int64_t rows = t.shape[0];
        const int64_t cols = t.shape[1];
        SB_THROW_IF(rows <= 0 || cols <= 0, "Operation '{}' requires a non-empty matrix, got {}x{}.", name, rows, cols);

        // The matrix is filled with N(0,1) before the sweep; see src/README.md.
        normal(t, 0.0, 1.0);

        // The shorter axis is orthonormalized, read through t.strides; see src/README.md.
        const bool by_rows = (rows <= cols);
        const int64_t count = by_rows ? rows : cols;
        const int64_t length = by_rows ? cols : rows;
        const int64_t elem_stride = by_rows ? t.strides[1] : t.strides[0];
        const int64_t vec_stride = by_rows ? t.strides[0] : t.strides[1];

        void* ptr = t.data();
        std::vector<SushiRuntime::Core::ResourceRegion> rw = {t.region()};

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = name;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.orthogonal"_op;
        meta.set_param(0, gain);

        engine_.get_graph().add_task(meta, rw, rw,
            [dtype = t.dtype, ptr, count, length, elem_stride, vec_stride, gain, name]
            (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                SB_LOG_INFO("RandomOps: {} ({} vectors of length {}, gain: {:.4f})", name, count, length, gain);

                switch (dtype)
                {
                    case Core::DataType::FLOAT32:
                        return orthogonal_dispatch(q, static_cast<float*>(ptr), count, length,
                                                   elem_stride, vec_stride, static_cast<float>(gain), deps);
                    case Core::DataType::FLOAT64:
                        return orthogonal_dispatch(q, static_cast<double*>(ptr), count, length,
                                                   elem_stride, vec_stride, gain, deps);
                    default:
                        SB_THROW_IF(true, "Operation '{}' does not support dtype {}.", name, Core::to_string(dtype));
                        return sycl::event();
                }
            }
        );

        return sycl::event();
    }
} // namespace SushiBLAS
