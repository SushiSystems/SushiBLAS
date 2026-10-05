/****************************************************************/
/* nrm2.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file nrm2.cpp
 * @brief Defines ReductionOps::nrm2, which records the Euclidean norm reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The norm is the scaled form of xNRM2, kept inside the accumulator; see src/README.md.
    sycl::event ReductionOps::nrm2(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::ScaledSumOfSquares>(
            engine_, t, result, "math.nrm2", "math.nrm2"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
