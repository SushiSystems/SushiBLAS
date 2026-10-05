/****************************************************************/
/* count_nonfinite.cpp                                          */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <cstdint>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // See the `@brief` on ReductionOps::count_nonfinite for why the result is an INT64
    // count rather than a flag in the input's dtype, and for why `asum` - which would
    // almost have served - is a false-positive generator on exactly the mixed-precision
    // input this exists for.
    //
    // The op itself is one call to the counting seam with a predicate transform. It gets
    // the library's own fixed-order two-stage fold from `Reduce::reduce` for free, which
    // is the point of putting the accumulation type in that function's signature rather
    // than hand-writing a second copy of the shape here.
    sycl::event ReductionOps::count_nonfinite(const Tensor& t, Tensor& result)
    {
        // Sized from the device the data lives on, exactly as execute_folding_reduction
        // does; Reduce::plan clamps to whatever scratch exists at launch, so a
        // disagreement costs performance and never correctness.
        const std::size_t partials =
            Reduce::planned_partials(Device::profile_for(t.get_device()),
                                     static_cast<std::size_t>(t.num_elements));

        auto dispatch = [](sycl::queue& q, const auto* pT, int64_t* pR, int64_t size,
                           auto scratch, const std::vector<sycl::event>& deps)
        {
            return Reduce::reduce<Reduce::ExactSum>(
                q, pT, pR, static_cast<std::size_t>(size),
                scratch.data, scratch.capacity,
                // `sycl::isfinite`, not a hand-rolled `x != x`. That spelling is the
                // classic NaN test and it is exactly half a non-finite test: an infinity
                // compares equal to itself, so `x != x` would wave every overflowed fp16
                // gradient straight through. `isfinite` is false for both, which is the
                // whole predicate in one call and nothing to get subtly wrong. Substituting
                // `x == x` here fails 13 of this op's 23 tests - every infinity case, at
                // every dtype, and none of the NaN ones - which is what makes the suite's
                // insistence on testing the two separately more than a stylistic point.
                [](auto x) -> int64_t { return sycl::isfinite(x) ? int64_t{0} : int64_t{1}; },
                Dispatch::ReduceIdentityFinalize{}, deps);
        };

        return Dispatch::execute_counting_reduction(engine_, t, result,
            "math.count_nonfinite", "math.count_nonfinite"_op, dispatch, partials);
    }
} // namespace SushiBLAS
