/****************************************************************/
/* sum.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file sum.cpp
 * @brief Defines ReductionOps::sum, which records the sum reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The sum is order-sensitive, so it goes through the folding seam; see src/README.md.
    sycl::event ReductionOps::sum(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
            engine_, t, result, "math.sum", "math.sum"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
