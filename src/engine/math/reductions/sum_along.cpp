/****************************************************************/
/* sum_along.cpp                                                */
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

    // Order-sensitive, so it takes the same Neumaier-compensated fixed-order fold
    // the whole-tensor `sum` takes - the segmented shape rather than the
    // whole-tensor one, and otherwise the same argument. Nothing to transform and
    // nothing to finalize; a per-axis `mean` is this call with `v / n` in the
    // finalizer slot and costs no extra launch.
    sycl::event ReductionOps::sum_along(const Tensor& t, int axis, Tensor& result)
    {
        return Dispatch::execute_folding_axis_reduction<Reduce::CompensatedSum>(
            engine_, t, axis, result, "math.sum_along", "math.sum_along"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
