/****************************************************************/
/* mean.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file mean.cpp
 * @brief Defines ReductionOps::mean, which records the mean reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The division by n is the finalizer of the fold; see src/README.md.
    sycl::event ReductionOps::mean(const Tensor& t, Tensor& result)
    {
        return Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
            engine_, t, result, "math.mean", "math.mean"_op,
            Dispatch::ReduceIdentityTransform{},
            [](auto value, std::size_t n)
            {
                using T = decltype(value);
                return value / static_cast<T>(n);
            });
    }
} // namespace SushiBLAS
