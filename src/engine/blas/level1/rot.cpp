/****************************************************************/
/* rot.cpp                                                      */
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

    sycl::event Level1::rot(Tensor& x, Tensor& y, double c, double s)
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in ROT.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "ROT requires tensors of the same number of elements.");

        // TODO: Implement multi-dimensional batch support

        void* rw_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        // Each new x_i is built from the old x_i and y_i (and vice versa), so both operands
        // are read as well as written.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (rw_x) reads.push_back(x.region());
        if (rw_y) reads.push_back(y.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_x) writes.push_back(x.region());
        if (rw_y) writes.push_back(y.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.rot";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.rot"_op;
        meta.set_param(0, c);
        meta.set_param(1, s);

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                using Real = typename Kernels::Reference::RealOf<T>::type;
                auto pX = static_cast<T*>(rw_x);
                auto pY = static_cast<T*>(rw_y);
                auto c_t = static_cast<Real>(c);
                auto s_t = static_cast<Real>(s);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::rot<T>(n, pX, incx, pY, incy, c_t, s_t); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::rot<T>(q2, n, pX, incx, pY, incy, c_t, s_t, deps2);
                    });
            });
    }
}
