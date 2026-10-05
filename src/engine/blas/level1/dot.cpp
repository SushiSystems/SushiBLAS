/****************************************************************/
/* dot.cpp                                                      */
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

    sycl::event Level1::dot(const Tensor& x, const Tensor& y, Tensor& result)
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in DOT.");
        // Note: for complex dtypes this computes the conjugated dot product (DOTC),
        // matching the previous oneMKL-backed behavior.

        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny;
        Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "DOT requires tensors of the same number of elements.");

        // TODO: Implement multi-dimensional batch support for Level 1 DOT

        void* read_x = x.data_or_null();
        void* read_y = y.data_or_null();
        void* write_r = result.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        if (read_y) reads.push_back(y.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_r) writes.push_back(result.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.dot";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.dot"_op;

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pY = static_cast<const T*>(read_y);
                auto pR = static_cast<T*>(write_r);

                if constexpr (Core::is_complex_v<T>)
                {
                    return Internal::dispatch_with_mode(mode, q, deps,
                        [=] { Kernels::Reference::dotc<typename T::value_type>(n, pX, incx, pY, incy, pR); },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::dotc<typename T::value_type>(q2, n, pX, incx, pY, incy, pR, deps2);
                        });
                }
                else
                {
                    return Internal::dispatch_with_mode(mode, q, deps,
                        [=] { Kernels::Reference::dot<T>(n, pX, incx, pY, incy, pR); },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::dot<T>(q2, n, pX, incx, pY, incy, pR, deps2);
                        });
                }
            });
    }
} // namespace SushiBLAS
