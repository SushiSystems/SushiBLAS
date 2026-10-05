/****************************************************************/
/* prod.cpp                                                     */
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

    // RunningProduct, not CompensatedSum: Neumaier recovers the bits an addition
    // rounds off and has nothing to say about multiplication. What this shares with
    // the additive reductions - and the whole reason it moved off sycl::reduction
    // too - is the fixed combination order, since floating-point multiplication is
    // no more associative than addition is.
    sycl::event ReductionOps::prod(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::RunningProduct>(
            engine_, t, result, "math.prod", "math.prod"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
