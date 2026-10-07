/**************************************************************************/
/* asum.cpp                                                               */
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
 * @file asum.cpp
 * @brief Defines Level1::asum, which records the BLAS ASUM task.
 * @author Mustafa Garip
 */

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level1::asum(const Tensor& x, Tensor& result)
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx, "blas.lvl1.asum", "x");
        Internal::require_scalar_result(result, Internal::real_result_dtype(x.dtype), "blas.lvl1.asum");

        const void* read_x = x.data_or_null();
        void* write_r = result.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_r) writes.push_back(result.region());

        const ScratchBuffer scratch = Internal::allocate_level1_scratch(
            engine_, x, static_cast<std::size_t>(n),
            Kernels::SYCL::K_LEVEL1_PARTIAL_BYTES, writes);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.asum";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.asum"_op;

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            Internal::ReferenceOperands{read_x, write_r},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                using Real = typename Kernels::Reference::RealOf<T>::type;
                auto pX = static_cast<const T*>(read_x);
                auto pR = static_cast<Real*>(write_r);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::asum<T>(n, pX, incx, pR); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::asum<T>(q2, n, pX, incx, pR, scratch.data(), scratch.bytes(), deps2);
                    });
            });
    }
}
