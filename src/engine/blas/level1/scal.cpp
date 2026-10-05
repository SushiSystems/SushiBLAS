/****************************************************************/
/* scal.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file scal.cpp
 * @brief Defines Level1::scal, which records the BLAS SCAL task.
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

    sycl::event Level1::scal(double alpha, Tensor& x)
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);

        void* rw_x = x.data_or_null();

        // In place: `x = alpha * x` reads every element it writes.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (rw_x) reads.push_back(x.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_x) writes.push_back(x.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.scal";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.scal"_op;
        meta.set_param(0, alpha);

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<T*>(rw_x);
                auto alpha_t = static_cast<T>(alpha);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::scal<T>(n, alpha_t, pX, incx); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::scal<T>(q2, n, alpha_t, pX, incx, deps2);
                    });
            });
    }
}
