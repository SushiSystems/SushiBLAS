/**************************************************************************/
/* copy.cpp                                                               */
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
 * @file copy.cpp
 * @brief Defines Level1::copy, which records the BLAS COPY task.
 * @author Mustafa Garip
 */

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level1::copy(const Tensor& x, Tensor& y)
    {
        // A dtype mismatch stays an error and names the converting copy; see src/README.md.
        SB_THROW_IF(x.dtype != y.dtype,
                    "Data type mismatch in COPY: source is {}, destination is {}. BLAS "
                    "copy moves elements without changing their type; for a copy that "
                    "converts, use elementwise().convert(src, dst).",
                    Core::to_string(x.dtype), Core::to_string(y.dtype));
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx, "blas.lvl1.copy", "x");
        int64_t ny; Internal::get_vec_params(y, ny, incy, "blas.lvl1.copy", "y");
        SB_THROW_IF(n != ny, "COPY requires tensors of the same number of elements.");

        const void* read_x = x.data_or_null();
        void* write_y = y.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_y) writes.push_back(y.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.copy";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.copy"_op;

        // copy does no arithmetic, so it serves Domain::ALL; see src/README.md.
        return Internal::execute_blas_task<Dispatch::Domain::ALL>(engine_, meta, reads, writes, x.dtype,
            Internal::ReferenceOperands{read_x, write_y},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<T*>(write_y);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::copy<T>(n, pX, incx, pY, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::copy<T>(q2, n, pX, incx, pY, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
