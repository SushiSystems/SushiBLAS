/****************************************************************/
/* max_along.cpp                                                */
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

    // Through the folding seam rather than sycl::reduction, which is the opposite
    // of what the whole-tensor `max` does one file over - and for a reason that is
    // about shape rather than about `maximum`. `sycl::reduction` reduces to a
    // scalar; there is no segmented form of it to reach for here at all. So the
    // axis family has one shape serving every accumulator, and `maximum` being
    // exactly associative simply means this one is bit-reproducible for a stronger
    // reason than the additive ones are. See Reduce::RunningMax.
    sycl::event ReductionOps::max_along(const Tensor& t, int axis, Tensor& result)
    {
        return Dispatch::execute_folding_axis_reduction<Reduce::RunningMax>(
            engine_, t, axis, result, "math.max_along", "math.max_along"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
