/****************************************************************/
/* iamax.cpp                                                    */
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

    // The result is an index, so the result tensor's dtype must be INT64 - and this
    // check is the whole point of the integer dtypes existing.
    //
    // Before them, `iamax` wrote an `int64_t` into whatever the caller passed, which in
    // every test and every consumer was a FLOAT32 tensor. Eight bytes into a four-byte
    // element. `tests/functional/unit/test_iamax.cpp` worked around it by allocating two
    // floats and left a comment reading "Wait, different sizes!". That was survivable
    // while a tensor owned its allocation; it stopped being survivable when SushiAI began
    // packing tensors into arenas, because the four bytes past the result belong to a
    // different tensor - and `Tensor::region()` reported a four-byte write, so the
    // dependency tracker ordered nothing against the corruption either.
    sycl::event Level1::iamax(const Tensor& x, Tensor& result)
    {
        SB_THROW_IF(result.dtype != Core::DataType::INT64,
                    "Operation 'blas.lvl1.iamax' writes a 64-bit index, so its result tensor "
                    "must have dtype INT64; got {}. A narrower dtype would be written past "
                    "the end of its element.",
                    Core::to_string(result.dtype));
        SB_THROW_IF(result.num_elements != 1,
                    "Result tensor for 'blas.lvl1.iamax' must be a scalar (1 element); got {}.",
                    result.num_elements);

        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);

        void* read_x = x.data_or_null();
        void* write_r = result.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_x) reads.push_back(x.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_r) writes.push_back(result.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "blas.lvl1.iamax";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.lvl1.iamax"_op;

        auto mode = engine_.get_mode();

        return Internal::execute_blas_task(engine_, meta, reads, writes, x.dtype,
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pX = static_cast<const T*>(read_x);
                auto pR = static_cast<int64_t*>(write_r);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::iamax<T>(n, pX, incx, pR); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::iamax<T>(q2, n, pX, incx, pR, deps2);
                    });
            });
    }
}
