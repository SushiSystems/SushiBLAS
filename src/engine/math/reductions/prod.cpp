/****************************************************************/
/* prod.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file prod.cpp
 * @brief Defines ReductionOps::prod, which records the product reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The product folds in a fixed order through RunningProduct; see src/README.md.
    sycl::event ReductionOps::prod(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::RunningProduct>(
            engine_, t, result, "math.prod", "math.prod"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
