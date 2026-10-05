/****************************************************************/
/* asum.cpp                                                     */
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

    sycl::event Level1::asum(const Tensor& x, Tensor& result)
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);

        void* read_x = x.data_or_null();
        void* write_r = result.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_r) writes.push_back(result.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.asum";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.asum"_op;

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                using Real = typename Kernels::Reference::RealOf<T>::type;
                auto pX = static_cast<const T*>(read_x);
                auto pR = static_cast<Real*>(write_r);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::asum<T>(n, pX, incx, pR); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::asum<T>(q2, n, pX, incx, pR, deps2);
                    });
            });
    }
}
