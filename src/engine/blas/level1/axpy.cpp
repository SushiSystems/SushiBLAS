/****************************************************************/
/* axpy.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <complex>
#include <kernels/blas.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/level1.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level1::axpy(double alpha, const Tensor& x, Tensor& y)
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in AXPY.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "AXPY requires tensors of the same number of elements.");

        // TODO: Implement multi-dimensional batch support for Level 1 AXPY

        void* read_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        // y is an accumulator, not a destination: `y = alpha * x + y` reads it before it
        // writes it, so it belongs in both sets.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        if (rw_y) reads.push_back(y.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_y) writes.push_back(y.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.axpy";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.axpy"_op;
        meta.set_param(0, alpha);

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<T*>(rw_y);
                auto alpha_t = static_cast<T>(alpha);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::axpy<T>(n, alpha_t, pX, incx, pY, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::axpy<T>(q2, n, alpha_t, pX, incx, pY, incy, deps2);
                    });
            });
    }
}
