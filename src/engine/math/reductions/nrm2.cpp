/****************************************************************/
/* nrm2.cpp                                                     */
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

    // The scaling 2-norm BLAS's xNRM2 specifies, not the naive `sqrt(sum(x*x))` this
    // used to be. There is no transform and no finalizer any more, because the
    // squaring and the square root are both things that have to happen *inside* a
    // running scale to be safe - see Reduce::ScaledSumOfSquares for the algorithm and
    // for why the accumulator, not the transform, is where this belongs. Still one
    // launch on the single-pass shape and two on the two-stage one, exactly as before.
    sycl::event ReductionOps::nrm2(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::ScaledSumOfSquares>(
            engine_, t, result, "math.nrm2", "math.nrm2"_op,
            Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
    }
} // namespace SushiBLAS
