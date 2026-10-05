/****************************************************************/
/* max_along.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file max_along.cpp
 * @brief Defines ReductionOps::max_along, which records the per-axis maximum reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // An axis reduction has no sycl::reduction form, so it folds; see src/README.md.
    sycl::event ReductionOps::max_along(const Tensor& t, int axis, Tensor& result)
    {
        return Dispatch::execute_folding_axis_reduction<Reduce::RunningMax>(
            engine_, t, axis, result, "math.max_along", "math.max_along"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
