/****************************************************************/
/* min.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file min.cpp
 * @brief Defines ReductionOps::min, which records the minimum reduction.
 * @author Mustafa Garip
 */

#include <limits>
#include <type_traits>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    // minimum is exact in any combination order, except against NaN; see src/README.md.
    sycl::event ReductionOps::min(const Tensor& t, Tensor& result)
    {
        auto dispatch = [](sycl::queue& q, const auto* pT, auto* pR, int64_t size,
                           [[maybe_unused]] auto scratch,
                           const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;

            T init_val = std::numeric_limits<T>::max();
            return q.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size), sycl::reduction(pR, init_val, sycl::minimum<T>(), sycl::property::reduction::initialize_to_identity{}), [=](sycl::id<1> idx, auto& reducer) 
                {
                    reducer.combine(pT[idx[0]]);
                });
            });
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.min", "math.min"_op, dispatch);
    }
} // namespace SushiBLAS
