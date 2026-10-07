/**************************************************************************/
/* count_nonfinite.cpp                                                    */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

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
    TaskHandle ReductionOps::count_nonfinite(const Tensor& t, Tensor& result)
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

        return Dispatch::execute_counting_reduction(recorder_, t, result,
            "math.count_nonfinite", "math.count_nonfinite"_op, dispatch, partials);
    }
} // namespace SushiBLAS
