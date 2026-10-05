/****************************************************************/
/* sum_along.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file sum_along.cpp
 * @brief Defines ReductionOps::sum_along, which records the per-axis sum reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The sum is order-sensitive, so it takes the segmented compensated fold; see src/README.md.
    sycl::event ReductionOps::sum_along(const Tensor& t, int axis, Tensor& result)
    {
        return Dispatch::execute_folding_axis_reduction<Reduce::CompensatedSum>(
            engine_, t, axis, result, "math.sum_along", "math.sum_along"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
