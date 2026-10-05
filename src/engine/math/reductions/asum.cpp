/****************************************************************/
/* asum.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file asum.cpp
 * @brief Defines ReductionOps::asum, which records the sum-of-magnitudes reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // Sum of magnitudes: the same fold as `sum` with `fabs` as the transform.
    sycl::event ReductionOps::asum(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
            engine_, t, result, "math.asum", "math.asum"_op,
            [](auto x) { return sycl::fabs(x); },
            Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
