/****************************************************************/
/* count_nonfinite.cpp                                          */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file count_nonfinite.cpp
 * @brief Defines ReductionOps::count_nonfinite, which records the non-finite count reduction.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The count goes through the counting seam with a predicate transform; see src/README.md.
    sycl::event ReductionOps::count_nonfinite(const Tensor& t, Tensor& result)
    {
        // The partial count is planned from the device the data lives on; see src/README.md.
        const std::size_t partials =
            Reduce::planned_partials(Device::profile_for(t.get_device()),
                                     static_cast<std::size_t>(t.num_elements));

        auto dispatch = [](sycl::queue& q, const auto* pT, int64_t* pR, int64_t size,
                           auto scratch, const std::vector<sycl::event>& deps)
        {
            return Reduce::reduce<Reduce::ExactSum>(
                q, pT, pR, static_cast<std::size_t>(size),
                scratch.data, scratch.capacity,
                // isfinite is false for NaN and for infinity alike; see src/README.md.
                [](auto x) -> int64_t { return sycl::isfinite(x) ? int64_t{0} : int64_t{1}; },
                Dispatch::ReduceIdentityFinalize{}, deps);
        };

        return Dispatch::execute_counting_reduction(engine_, t, result,
            "math.count_nonfinite", "math.count_nonfinite"_op, dispatch, partials);
    }
} // namespace SushiBLAS
