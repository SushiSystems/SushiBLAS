/**************************************************************************/
/* rot.cpp                                                                */
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
 * @file rot.cpp
 * @brief Defines Level1::rot, which records the BLAS ROT task.
 * @author Mustafa Garip
 */

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle Level1::rot(Tensor& x, Tensor& y, double c, double s)
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in ROT.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx, "blas.lvl1.rot", "x");
        int64_t ny; Internal::get_vec_params(y, ny, incy, "blas.lvl1.rot", "y");
        SB_THROW_IF(n != ny, "ROT requires tensors of the same number of elements.");

        void* rw_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        // x and y are each read and written, so both sit in both access sets; see src/README.md.
        OperandSet operands;
        operands.updates(x).updates(y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.rot";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.rot"_op;
        meta.set_param(0, c);
        meta.set_param(1, s);

        return Internal::execute_blas_task(recorder_, meta, operands, x.dtype,
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                using Real = typename Kernels::Reference::RealOf<T>::type;
                auto pX = static_cast<T*>(rw_x);
                auto pY = static_cast<T*>(rw_y);
                auto c_t = static_cast<Real>(c);
                auto s_t = static_cast<Real>(s);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::rot<T>(n, pX, incx, pY, incy, c_t, s_t); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::rot<T>(q2, n, pX, incx, pY, incy, c_t, s_t, deps2);
                    });
            });
    }
}
