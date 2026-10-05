/****************************************************************/
/* sum.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // Order-sensitive, so it goes through the folding seam rather than
    // sycl::reduction - see <SushiBLAS/engine/reduce_1d.hpp> for why the spec's
    // unspecified combination order is a correctness problem for `+` and not for
    // `max`. Nothing to transform and nothing to finalize: this is the shape every
    // other reduction in this file is a variation on.
    sycl::event ReductionOps::sum(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
            engine_, t, result, "math.sum", "math.sum"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
