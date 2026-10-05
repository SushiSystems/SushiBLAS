/****************************************************************/
/* mean_spatial.cpp                                             */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file mean_spatial.cpp
 * @brief Defines ReductionOps::mean_spatial, which records the spatial mean reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The fold is the one mean uses, over a span keyed by (n, c); see src/README.md.
    sycl::event ReductionOps::mean_spatial(const Tensor& t, Tensor& result)
    {
        const Reduce::AxisSpan span =
            Dispatch::spatial_span_for(t, result, "math.mean_spatial");

        return Dispatch::execute_folding_span_reduction<Reduce::CompensatedSum>(
            engine_, t, span, result, "math.mean_spatial", "math.mean_spatial"_op,
            Dispatch::ReduceIdentityTransform{},
            [](auto value, std::size_t n)
            {
                using T = decltype(value);
                return value / static_cast<T>(n);
            });
    }
} // namespace SushiBLAS
