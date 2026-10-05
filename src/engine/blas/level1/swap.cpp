/****************************************************************/
/* swap.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file swap.cpp
 * @brief Defines Level1::swap, which records the BLAS SWAP task.
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

    sycl::event Level1::swap(Tensor& x, Tensor& y)
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in SWAP.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "SWAP requires tensors of the same number of elements.");

        void* rw_x = x.data_or_null();
        void* rw_y = y.data_or_null();

        // Both operands are read and written - a swap is the definition of read-modify-write.
        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (rw_x) reads.push_back(x.region());
        if (rw_y) reads.push_back(y.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (rw_x) writes.push_back(x.region());
        if (rw_y) writes.push_back(y.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.swap";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.swap"_op;

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<T*>(rw_x);
                auto pY = static_cast<T*>(rw_y);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::swap<T>(n, pX, incx, pY, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::swap<T>(q2, n, pX, incx, pY, incy, deps2);
                    });
            });
    }
}
