/****************************************************************/
/* copy.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "COPY requires tensors of the same number of elements.");

        void* read_x = x.data_or_null();
        void* write_y = y.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_y) writes.push_back(y.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.copy";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.copy"_op;

        auto mode = engine_.get_mode();

        // copy does no arithmetic, so it serves Domain::ALL; see src/README.md.
        return Internal::execute_blas_task<Dispatch::Domain::ALL>(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<T*>(write_y);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::copy<T>(n, pX, incx, pY, incy); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::copy<T>(q2, n, pX, incx, pY, incy, deps2);
                    });
            });
    }
} // namespace SushiBLAS
